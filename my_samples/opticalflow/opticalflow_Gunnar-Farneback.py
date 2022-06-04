import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('./sample.mov')

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 640.0

print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 360.0

print(cap.get(cv2.CAP_PROP_FPS))
# 29.97002997002997

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 360.0


frames = []
counter = 0
while True:
    ret, frame = cap.read()
    if ret:
        if (430 <= counter <= 440):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        counter += 1
    else:
        break


img_pre = frames[0].copy()
img_now = frames[4].copy()

img_pre_g = cv2.cvtColor(img_pre, cv2.COLOR_RGB2GRAY)
img_now_g = cv2.cvtColor(img_now, cv2.COLOR_RGB2GRAY)

rows, cols, ch = img_pre.shape

ps = np.empty((0, 2), np.float32)

FLOW_H = 10
FLOW_W = 10

flow = cv2.calcOpticalFlowFarneback(
    img_pre_g, img_now_g, None, 0.5, 3, 30, 3, 3, 1.1, 0)
rows, cols, ch = img_pre.shape

for y in range(0, cols, FLOW_H):
    for x in range(0, rows, FLOW_W):
        ps = (y, x)
        pe = (ps[0] + int(flow[x][y][0]), ps[1] + int(flow[x][y][1]))
        cv2.line(img_now, ps, pe, (255, 255, 255), 2)

plt.figure(figsize=(5, 5))
plt.imshow(img_now)
plt.show()
