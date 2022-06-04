import cv2
import numpy as np


def draw_flow(img, gray, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step /
                    2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T * 10
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # vis = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # グレー画像を背景に使用
    vis = 255 - vis  # ネガ反転
    rad = int(step / 2)

    i = 0  # ループカウンタ
    for (x1, y1), (x2, y2) in lines:
        pv = img[y1, x1]
        col = (int(pv[0]), int(pv[1]), int(pv[2]))
        r = rad - int(rad * abs(fx[i]) * .05)  # ドットの半径を移動ベクトルの量に応じて小さくする
        cv2.circle(vis, (x1, y1), abs(r), col, -1)
        i += 1
    cv2.polylines(vis, lines, False, (255, 255, 0), 2)
    return vis


cap = cv2.VideoCapture('./sample.mov')
# cap = cv2.VideoCapture('./sample2.mp4')
ret, frame = cap.read()
prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    # カメラのフレームをキャプチャ
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow[縦軸データ, 横軸データ, (x,y)]
    # print flow[0,0,:] # ＠ x,y ともに 0 番目のベクトルの大きさ, 角度（であってる？）
    prevgray = gray

    # プレビュー
    cv2.imshow('detect preview', draw_flow(frame, gray, flow, 32))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

    cv2.destroyAllWindows()
