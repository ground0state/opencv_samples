import cv2
import numpy as np

# file_src = 'src_depth_wall.png'
file_src = 'src_depth_hand.png'
file_dst = 'dst.png'

img_src = cv2.imread(file_src, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('src')
cv2.namedWindow('dst')

rows, cols = img_src.shape
img_255 = 255 * np.ones([rows, cols]).astype('uint8')

img_h = img_src
img_s = img_255
img_v = img_src

img_hsv = cv2.merge((img_h, img_s, img_v))
img_dst = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR_FULL)

cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.imwrite(file_dst, img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
