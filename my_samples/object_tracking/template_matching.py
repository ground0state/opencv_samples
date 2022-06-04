import cv2

img_src = cv2.imread('04-02-a.jpg')
img_template = cv2.imread('04-02-b.jpg')

# (W, H)
print('a shape:', img_src.shape)
# (w, h)
print('b shape:', img_template.shape)

img_dst = img_src.copy()

h, w, ch = img_template.shape
# 走査した引き算の結果 (W-w+1, H-h+1)
img_minmax = cv2.matchTemplate(img_src, img_template, cv2.TM_CCOEFF_NORMED)
# 値が最小と最大になる点とその値
min_val, max_val, min_pt, max_pt = cv2.minMaxLoc(img_minmax)

cv2.rectangle(
    img=img_dst,
    pt1=max_pt,
    pt2=(max_pt[0] + w, max_pt[1] + h),
    color=(255, 255, 255),
    thickness=10)

cv2.imshow('src', img_src)
cv2.imshow('template', img_template)
cv2.imshow('minmax', img_minmax)
cv2.imshow('dst', img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
