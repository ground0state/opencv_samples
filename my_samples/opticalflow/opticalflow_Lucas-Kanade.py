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


rows, cols = img_pre_g.shape

ps = np.empty((0, 2), np.float32)

FLOW_H = 10
FLOW_W = 10

for y in range(0, cols, FLOW_H):
    for x in range(0, rows, FLOW_W):
        pp = np.array([[y, x]], np.float32)
        ps = np.vstack([ps, pp])

pe, status, error = cv2.calcOpticalFlowPyrLK(img_pre_g, img_now_g, ps, None)


for i in range(len(ps)):
    cv2.line(
        img_now,
        (int(ps[i][0]), int(ps[i][1])),
        (int(pe[i][0]), int(pe[i][1])),
        color=(255, 255, 255),
        thickness=2
    )


plt.figure(figsize=(10, 5))
plt.imshow(img_now)
plt.show()
