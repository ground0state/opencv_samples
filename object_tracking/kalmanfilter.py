import cv2
import numpy as np
import time

H_UPPER = 115
H_LOWER = 60
S_UPPER = 255
S_LOWER = 50
V_UPPER = 200
V_LOWER = 20

WIN_KF = 'kf'
RESIZE = (640, 360)

PREDICTION_AHEAD = False
PREDICTION = True

# 状態変数(4)、観測値(2) の各サイズを指定してインスタンス作成
kf = cv2.KalmanFilter(4, 2, type=cv2.CV_64F)
# カルマンフィルタの設定をする
dt = 1  # データの時刻差は1単位時間とする
kf.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)
kf.measurementMatrix = np.eye(2, 4, dtype=np.float64)
kf.processNoiseCov = 0.1 * np.eye(4, 4, dtype=np.float64)
kf.measurementNoiseCov = 0.1 * np.eye(2, 2, dtype=np.float64)
# kf.errorCovPost = 0.1 * np.eye(4, 4)

# 初期値を与える
kf.statePost = np.array([[0, 0, 0, 0]], dtype=np.float64).T

# ノイズ除去に使用するフィルター
element8 = np.eye(3, 3, dtype=np.uint8)

cap = cv2.VideoCapture('04-06.wmv')
cv2.namedWindow(WIN_KF, cv2.WINDOW_NORMAL)

while(True):
    ret, img_src = cap.read()

    if not ret:
        break

    img_src = cv2.resize(img_src, RESIZE)

    # 追跡対象抽出
    gray = cv2.cvtColor(img_src, cv2.COLOR_BGRA2GRAY)
    hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV_FULL)

    # HSV閾値処理
    img_bin = cv2.inRange(
        hsv,
        (H_LOWER, S_LOWER, V_LOWER),
        (H_UPPER, S_UPPER, V_UPPER)
    )

    # ノイズ処理
    img_bin = cv2.erode(
        src=img_bin,
        kernel=element8,
        anchor=(-1, -1),
        iterations=5,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    img_bin = cv2.dilate(
        src=img_bin,
        kernel=element8,
        anchor=(-1, -1),
        iterations=5,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 面積最大ラベルの選択
    # nlabels: ラベルの数。各オブジェクトに番号がラベリングされており、nlabels はそのラベルの数を表す。ただし、画像の背景の番号は 0 とラベリングされているので、実際のオブジェクトの数は nlabels - 1 となる。
    # labels: 画像のラベリング結果を保持している二次元配列。配列の要素は、各ピクセルのラベル番号となっている。
    # stats: オブジェクトのバウンディングボックス（開始点の x 座標、y 座標、幅、高さ）とオブジェクトのサイズ。
    # centroids: オブジェクトの重心。
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)

    if nlabels == 1:  # 背景のみの場合
        continue

    max_area, max_index = 0, 0
    for i in range(1, nlabels):
        area = stats[i][4]
        if area > max_area:
            max_area = area
            max_index = i

    # 面積最大ラベルの重心
    pos = centroids[max_index]
    # 観測
    measurement = pos

    if PREDICTION_AHEAD:
        """1手先を予測
        """
        # 更新
        kf.correct(measurement)
        # 予測
        prediction = kf.predict().copy()
    elif PREDICTION:
        """現在位置を予測
        """
        # 予測
        prediction = kf.predict().copy()
        # 更新
        kf.correct(measurement)
    else:
        """現在位置を補正
        """
        prediction = kf.correct(measurement).copy()
        kf.predict()

    # 結果の描画 重心位置
    img_src = cv2.circle(
        img=img_src,
        center=np.round(pos).astype('int'),
        radius=3,
        color=(0, 0, 255),
        thickness=-1)
    # 結果の描画 予測位置
    img_src = cv2.circle(
        img=img_src,
        center=(int(prediction[0]), int(prediction[1])),
        radius=3,
        color=(0, 255, 0),
        thickness=-1)
    img_src = cv2.ellipse(
        img_src,
        (int(prediction[0]), int(prediction[1])),
        (int(abs(prediction[2])), int(abs(prediction[3]))),
        angle=0.0,
        startAngle=0.0,
        endAngle=360.0,
        color=(0, 255, 0),
        thickness=1
    )

    cv2.imshow(WIN_KF, img_src)

    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
