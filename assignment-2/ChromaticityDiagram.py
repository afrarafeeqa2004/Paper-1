import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH ="/content/drive/MyDrive/Colab Notebooks/25-jackson-hole-getty.jpg"
MAX_SAMPLES = 200000
def srgb_to_linear(rgb):
    rgb = np.asarray(rgb)
    a = 0.055
    linear = np.where(rgb <= 0.04045,
                      rgb / 12.92,
                      ((rgb + a) / (1 + a)) ** 2.4)
    return linear

def linear_rgb_to_xyz(rgb_linear):
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    shape = rgb_linear.shape
    flat = rgb_linear.reshape(-1, 3)
    xyz = flat.dot(M.T)
    return xyz.reshape(shape)

def xyz_to_xy(xyz):
    X = xyz[..., 0]
    Y = xyz[..., 1]
    Z = xyz[..., 2]
    denom = (X + Y + Z)
    small = denom == 0
    denom[small] = 1.0
    x = X / denom
    y = Y / denom
    x[small] = 0.0
    y[small] = 0.0
    return np.stack([x, y], axis=-1)

def load_image_as_rgb(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype("float32") / 255.0  # in 0..1
    return arr

img = load_image_as_rgb(IMAGE_PATH)
h, w, _ = img.shape
pixels = img.reshape(-1, 3)
n_pixels = pixels.shape[0]
if n_pixels > MAX_SAMPLES:
    idx = np.random.choice(n_pixels, size=MAX_SAMPLES, replace=False)
    pixels = pixels[idx]
pixels_linear = srgb_to_linear(pixels)
pixels_xyz = linear_rgb_to_xyz(pixels_linear)
pixels_xy = xyz_to_xy(pixels_xyz)  # shape (N,2)

def plot_xy_scatter_on_cie(pixels_xy, pixels_rgb, use_colour_if_available=True):
    plt.figure(figsize=(8,7))
    try:
        if use_colour_if_available:
            import colour
            from colour.plotting import plot_chromaticity_diagram_CIE1931
            ax = plt.gca()
            plot_chromaticity_diagram_CIE1931(standalone=False, axes=ax)
            xy = np.clip(pixels_xy, 0, 1)
            plt.scatter(xy[:,0], xy[:,1], s=1, c=pixels_rgb, marker='.', alpha=0.6)
            plt.title("Pixel chromaticities on CIE 1931 diagram")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim(-0.05, 0.8)
            plt.ylim(-0.05, 0.9)
            plt.grid(False)
            plt.show()
            return
    except Exception as e:
        print("colour-science not available or failed.", e)
    xy = pixels_xy
    plt.scatter(xy[:,0], xy[:,1], s=1, c=pixels_rgb, marker='.', alpha=0.6)
    srgb_primaries = np.array([[1,0,0],[0,1,0],[0,0,1]])  # R,G,B
    primaries_linear = srgb_primaries  # are already linear (1 or 0)
    prim_xyz = linear_rgb_to_xyz(primaries_linear)
    prim_xy = xyz_to_xy(prim_xyz).reshape(-1,2)
    white_linear = np.array([1.0,1.0,1.0])
    white_xy = xyz_to_xy(linear_rgb_to_xyz(white_linear.reshape(1,3))).reshape(2)
    tri = np.vstack([prim_xy, prim_xy[0]])
    plt.plot(tri[:,0], tri[:,1], '-', color='k', lw=1.5, label='sRGB gamut')
    # plot white point
    plt.plot(white_xy[0], white_xy[1], 'o', color='k', markersize=4)
    plt.text(white_xy[0]+0.01, white_xy[1], 'D65 (white)', fontsize=9)
    plt.title("Pixel chromaticities (CIE 1931)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-0.05, 0.8)
    plt.ylim(-0.05, 0.9)
    plt.legend()
    plt.grid(True)
    plt.show()
plot_xy_scatter_on_cie(pixels_xy, pixels)
