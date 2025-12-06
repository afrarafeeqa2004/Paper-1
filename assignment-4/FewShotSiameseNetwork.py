!pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import numpy as np
import random

#triplet Loss
def triplet_loss(y_true, y_pred, margin=1.0):
    anchor, positive, negative = tf.unstack(y_pred, num=3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)

#triplet generator — fixed output format
class TripletGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=32):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(np.unique(labels))
        self.class_indices = {i: np.where(labels == i)[0] for i in range(self.num_classes)}
        self.indices = np.arange(len(images))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.images) // self.batch_size

def __getitem__(self, idx):
        anchors, positives, negatives = [], [], []
        for _ in range(self.batch_size):
            anchor_idx = random.choice(self.indices)
            anchor_img = self.images[anchor_idx]
            anchor_label = self.labels[anchor_idx]

            # positive sample
            pos_idx = random.choice(self.class_indices[anchor_label])
            while pos_idx == anchor_idx:
                pos_idx = random.choice(self.class_indices[anchor_label])
            positive_img = self.images[pos_idx]

            # negative sample
            neg_label = random.choice([l for l in range(self.num_classes) if l != anchor_label])
            neg_idx = random.choice(self.class_indices[neg_label])
            negative_img = self.images[neg_idx]

            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)

        anchors = np.array(anchors, dtype="float32")
        positives = np.array(positives, dtype="float32")
        negatives = np.array(negatives, dtype="float32")
        y_dummy = np.zeros((self.batch_size,), dtype="float32")

        return (anchors, positives, negatives), y_dummy

#siamese model definition
def build_siamese_model(input_shape, embedding_dim=128):
    base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    #base_cnn.trainable = False
    for layer in base_cnn.layers[-30:]:  # unfreeze last 30 layers
      layer.trainable = True
    input_tensor = keras.Input(shape=input_shape)
    x = base_cnn(input_tensor, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim, activation="relu")(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    embedding_network = keras.Model(input_tensor, x, name="embedding_network")

    input_a = keras.Input(shape=input_shape, name="anchor")
    input_p = keras.Input(shape=input_shape, name="positive")
    input_n = keras.Input(shape=input_shape, name="negative")

    emb_a = embedding_network(input_a)
    emb_p = embedding_network(input_p)
    emb_n = embedding_network(input_n)

    #Lambda layer to safely stack tensors
    output = layers.Lambda(lambda tensors: tf.stack(tensors, axis=1))([emb_a, emb_p, emb_n])

    siamese_model = keras.Model(inputs=[input_a, input_p, input_n], outputs=output)
    return siamese_model, embedding_network

#main training
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, y_train = x_train[:5000], y_train[:5000]

    input_shape = (32, 32, 3)
    train_gen = TripletGenerator(x_train, y_train, batch_size=32)
    val_gen = TripletGenerator(x_test[:1000], y_test[:1000], batch_size=32)
    # 156 batches with 5000 images and 32 batch size
    siamese_model, embedding_network = build_siamese_model(input_shape)
    siamese_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=triplet_loss)

    siamese_model.fit(train_gen, validation_data=val_gen, epochs=2)

    #similarity check
    img1 = x_test[0:1]
    img2_same = x_test[1:2]
    img3_diff = x_test[10:11]

    emb1 = embedding_network.predict(img1)
    emb2 = embedding_network.predict(img2_same)
    emb3 = embedding_network.predict(img3_diff)

    cosine = tf.keras.losses.cosine_similarity
    print("\nCosine Similarity (same class):", cosine(emb1, emb2).numpy())
    print("Cosine Similarity (diff class):", cosine(emb1, emb3).numpy())
