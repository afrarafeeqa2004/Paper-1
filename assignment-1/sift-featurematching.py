import cv2
import numpy as np
import matplotlib.pyplot as plt

scene = cv2.imread("product.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("logo.png", cv2.IMREAD_GRAYSCALE)
h, w = template.shape[:2]

res = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

output_tm = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
if max_val > 0.6: 
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(output_tm, top_left, bottom_right, (0,0,255), 3)
    cv2.putText(output_tm, f"Score: {max_val:.2f}", (top_left[0], top_left[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
else:
    cv2.putText(output_tm, "No reliable match", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#ORB Feature Matching
orb = cv2.ORB_create(2000)
kp1_orb, des1_orb = orb.detectAndCompute(template, None)
kp2_orb, des2_orb = orb.detectAndCompute(scene, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

output_orb = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
if len(matches_orb) > 10:
    src_pts = np.float32([kp1_orb[m.queryIdx].pt for m in matches_orb]).reshape(-1,1,2)
    dst_pts = np.float32([kp2_orb[m.trainIdx].pt for m in matches_orb]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is not None:
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(output_orb, [np.int32(dst)], True, (0,255,0), 3)
        cv2.putText(output_orb, "ORB Match", (30,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#SIFT Feature Matching
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(template, None)
kp2_sift, des2_sift = sift.detectAndCompute(scene, None)

bf_sift = cv2.BFMatcher()
matches_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

good_sift = []
for m, n in matches_sift:
    if m.distance < 0.75 * n.distance:
        good_sift.append(m)

output_sift = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
if len(good_sift) > 10:
    src_pts = np.float32([kp1_sift[m.queryIdx].pt for m in good_sift]).reshape(-1,1,2)
    dst_pts = np.float32([kp2_sift[m.trainIdx].pt for m in good_sift]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is not None:
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(output_sift, [np.int32(dst)], True, (255,0,0), 3)
        cv2.putText(output_sift, "SIFT Match", (30,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(output_tm[:,:,::-1])
plt.title("Template Matching (Rigid)")

plt.subplot(1,3,2)
plt.imshow(output_orb[:,:,::-1])
plt.title("ORB Feature Matching (Rotation/Scale Invariant)")

plt.subplot(1,3,3)
plt.imshow(output_sift[:,:,::-1])
plt.title("SIFT Feature Matching (Robust)")

plt.show()
