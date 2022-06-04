import cv2
import numpy as np
import sys

best = 30


def main():
    file_cam_param = "./cam_param.xml"
    filename = ["./regist5-1.jpg", "./regist5-2.jpg"]
    color = [(0, 0, 255), (255, 0, 0)]

    img_src = [None] * 2
    img_srcw = [None] * 2
    for i in range(2):
        img_src[i] = cv2.imread(filename[i], flags=cv2.IMREAD_COLOR)
        cv2.rectangle(img_src[i],
                      pt1=(0, 0),
                      pt2=img_src[i].shape[1::-1],
                      color=color[i],
                      thickness=2
                      )
        img_srcw[i] = np.zeros((2 * img_src[i].shape[0], 2 * img_src[i].shape[1], 3), dtype=img_src[i].dtype)
        img_srcw[i][int(img_srcw[i].shape[0] / 4):int(img_srcw[i].shape[0] / 4 * 3),
                    int(img_srcw[i].shape[1] / 4):int(img_srcw[i].shape[1] / 4 * 3)] = img_src[i]

    cv2.imshow("img_src[0]", img_srcw[0])
    cv2.imshow("img_src[1]", img_srcw[1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 特徴点抽出
    detector = cv2.AKAZE_create()
    kpts1, desc1 = detector.detectAndCompute(img_srcw[0], None)
    kpts2, desc2 = detector.detectAndCompute(img_srcw[1], None)

    if len(kpts1) < best or len(kpts2) < best:
        print("few keypoints")
        sys.exit()

    # 得られた特徴点間のマッチング
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    print(f"best = {best}")
    print(f"match size = {len(matches)}")
    if len(matches) < best:
        print("few matchpoints")

    # 上位best個を採用
    matches = sorted(matches, key=lambda x: x.distance)[:best]

    # 特徴点の対応を表示
    img_match = cv2.drawMatches(img_srcw[0], kpts1, img_srcw[1], kpts2, matches, None, flags=2)
    cv2.imshow("matchs", img_match)

    # 特徴点をvectorにまとめる
    points_src = []
    points_dst = []
    for i in range(len(matches)):
        points_src.append(kpts1[matches[i].queryIdx].pt)
        points_dst.append(kpts2[matches[i].trainIdx].pt)
    points_src = np.array(points_src)
    points_dst = np.array(points_dst)

    # マッチング結果から，F行列を推定する
    F = cv2.findFundamentalMat(
        **{"points1": points_src,
           "points2": points_dst,
           "method": cv2.FM_RANSAC,
           "ransacReprojThreshold": 3,
           "confidence": 0.99,
           "maxIters": 1000}
    )
    print(f"F = {F}")

    # カメラの内部パラメータが既知の場合はE行列を計算し，外部パラメータを推定する
    # カメラ内部パラメータ読み込み
    f = cv2.FileStorage(file_cam_param, flags=cv2.FILE_STORAGE_READ)
    A = f.getNode('intrinsic').mat()
    f.release()
    print(f"A = {A}")

    # E行列の計算
    E, mask_match = cv2.findEssentialMat(points_src, points_dst, A)

    # 外部パラメータ（回転，並進ベクトル）の計算
    retval, R, t, mask = cv2.recoverPose(E, points_src, points_dst, A)

    print(f"R = {R}")
    print(f"t = {t}")

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
