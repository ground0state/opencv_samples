import cv2
import numpy as np


file_src = 'src_depth_ball.png'
file_dst = 'dst.png'

img_src = cv2.imread(file_src, 0)

cv2.namedWindow('src')
cv2.namedWindow('dst')

depth_min = 206
depth_max = 210
img_dst = cv2.inRange(img_src, depth_min, depth_max)

kernel = np.ones((3, 3), np.uint8)
img_dst = cv2.erode(img_dst, kernel, iterations=2)
img_dst = cv2.dilate(img_dst, kernel, iterations=2)

nlabel, img_lab = cv2.connectedComponents(img_dst)
for i in range(1, nlabel, 1):
    img_dst = cv2.compare(img_lab, i, cv2.CMP_EQ)
    print(i, ' / ', (nlabel - 1))
    cv2.imwrite('dst' + str(i) + '.png', img_dst)
    cv2.imshow('src', img_src)
    cv2.imshow('dst', img_dst)
    cv2.waitKey(0)

cv2.destroyAllWindows()


file_src = 'src_depth_ball.png'
file_dst = 'dst.png'

img_src = cv2.imread(file_src, 0)

depth_min = 206
depth_max = 210
img_dst = cv2.inRange(img_src, depth_min, depth_max)

kernel = np.ones((3, 3), np.uint8)
img_dst = cv2.erode(img_dst, kernel, iterations=2)
img_dst = cv2.dilate(img_dst, kernel, iterations=2)

nlabel, img_lab = cv2.connectedComponents(img_dst)
