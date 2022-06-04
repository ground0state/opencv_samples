import cv2
import numpy as np

RESIZE = (640, 360)

win_src = 'src'
win_bin = 'bin'

cap = cv2.VideoCapture('04-06.wmv')
ret, img_src = cap.read()
img_src = cv2.resize(img_src, RESIZE)
h, w, ch = img_src.shape

div = 6
# 初期位置
rct = (1 * int(w / div), 1 * int(h / div), int(w / div), int(h / div))

cv2.namedWindow(win_bin, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_src, cv2.WINDOW_NORMAL)

# termination criteria
cri = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

# ノイズ除去に使用するフィルター
element8 = np.eye(3, 3, dtype=np.uint8)

while(True):
    ret, img_src = cap.read()

    if not ret:
        break

    img_src = cv2.resize(img_src, RESIZE)

    # BGRだけで二値化
    # bgr = cv2.split(img_src)
    # th = 220
    # ret, img_bin = cv2.threshold(bgr[2], th, 255, cv2.THRESH_BINARY)

    # 色指定で二値化
    hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 50, 20])
    hsv_max = np.array([100, 255, 200])
    img_bin = cv2.inRange(hsv, hsv_min, hsv_max)

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

    # meanshift実行
    ret, rct = cv2.meanShift(img_bin, rct, cri)
    x, y, w, h = rct
    img_src = cv2.rectangle(img_src, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow(win_src, img_src)
    cv2.imshow(win_bin, img_bin)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
