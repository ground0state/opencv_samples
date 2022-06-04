import cv2

img_path_list = [
    "05-08-a.jpg",
    "05-08-b.jpg",
    "05-08-c.jpg",
    "05-08-d.jpg",
    "05-08-e.jpg",
    "05-08-f.jpg",
]

input_images = []
for i in img_path_list:
    image = cv2.imread(i)
    if image is None:
        print(f'Error: Unable to open file "{i}".')
        exit()
    input_images.append(image)

if len(input_images) <= 1:
    raise

stitcher = cv2.Stitcher_create()
stitched = stitcher.stitch(input_images)

cv2.imshow('dst', stitched[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
