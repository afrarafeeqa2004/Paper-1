import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("E:/Afra/datasets/archive/hazelnut/test/print/image-4.png", cv2.IMREAD_GRAYSCALE)  # reference
img2 = cv2.imread("E:/Afra/datasets/archive/hazelnut/test/print/008.png", cv2.IMREAD_GRAYSCALE)      # test/scene

#SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#ORB
orb = cv2.ORB_create(5000)
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)

#matching-SIFT with FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_sift = flann.knnMatch(des1, des2, k=2)

good_sift = []
for m, n in matches_sift:
    if m.distance < 0.7 * n.distance:
        good_sift.append(m)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

sift_result = cv2.drawMatches(img1, kp1, img2, kp2, good_sift[:50], None, flags=2)
orb_result = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, matches_orb[:50], None, flags=2)

cv2.imwrite("sift_matches.png", sift_result)
cv2.imwrite("orb_matches.png", orb_result)

if len(good_sift) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    img2_aligned = cv2.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    cv2.imwrite("aligned.png", img2_aligned)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1); plt.imshow(sift_result, cmap='gray'); plt.title("sift image"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(orb_result, cmap='gray'); plt.title("orb images"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(img2_aligned, cmap='gray'); plt.title("aligned image"); plt.axis('off')
plt.show()
