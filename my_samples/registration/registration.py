import cv2
import numpy as np

BEST = 40

# 2枚まで対応
img_path_list = [
    "05-06-a.jpg",
    "05-06-b.jpg"
]


# 画像読み込み
img_src_list = []
img_srcw_list = []
for img_path in img_path_list:
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, c = img.shape
    img_rec = cv2.rectangle(
        img.copy(),
        pt1=(0, 0),
        pt2=(w, h),
        color=(0, 255, 0),
        thickness=2)

    img_g = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    img_src = np.zeros((2 * h, 2 * w, c), dtype=np.uint8)
    img_srcw = np.zeros((2 * h, 2 * w), dtype=np.uint8)

    img_src[int(h / 4):int(h / 4) + h, int(w / 4):int(w / 4) + w] = img_rec
    img_srcw[int(h / 4):int(h / 4) + h, int(w / 4):int(w / 4) + w] = img_g

    img_src_list.append(img_src)
    img_srcw_list.append(img_srcw)

if len(img_srcw_list) == 0:
    raise


# Create ORB detector with 5000 features.
orb_detector = cv2.ORB_create(5000)

res_list = []
for img_srcw in img_srcw_list:
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp, des = orb_detector.detectAndCompute(img_srcw, None)

    if len(kp) < BEST:
        # 特徴点が少なすぎる場合は停止
        raise

    res_list.append(
        dict(kp=kp, des=des)
    )


# Match features between the two images.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
for j, (res1, res2) in enumerate(zip(res_list[:-1], res_list[1:])):
    # Match the two sets of descriptors.
    matches = list(matcher.match(res1['des'], res2['des']))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top matches forward.
    matches = matches[:BEST]
    no_of_matches = len(matches)

    h1, w1, _ = img_src_list[j].shape
    h2, w2, _ = img_src_list[j + 1].shape
    img_dst = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    cv2.drawMatches(img_src_list[j], res1['kp'], img_src_list[j + 1], res2['kp'], matches, img_dst)
    cv2.imshow('dst', img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = res1['kp'][matches[i].queryIdx].pt
        p2[i, :] = res2['kp'][matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    h, w, _ = img_src_list[j].shape
    transformed_img = cv2.warpPerspective(
        img_src_list[j],
        homography, (w, h))

    img_reg = cv2.addWeighted(
        src1=img_src_list[j + 1],
        alpha=0.5,
        src2=transformed_img,
        beta=0.5,
        gamma=0.0)

cv2.imshow('dst', img_reg)
cv2.waitKey(0)
cv2.destroyAllWindows()
