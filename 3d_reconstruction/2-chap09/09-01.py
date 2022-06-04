import cv2
import numpy as np
import sys

windowDisparity = "Disparity"
fileLeft = "./09-04-a.png"
fileRight = "./09-04-b.png"


def main():
    # 画像ファイルの読み込み
    imgLeft = cv2.imread(fileLeft, cv2.IMREAD_GRAYSCALE)
    imgRight = cv2.imread(fileRight, cv2.IMREAD_GRAYSCALE)
    if imgLeft is None or imgRight is None:
        print(" --指定されたファイルがありません！")
        sys.exit()

    # 探索したいdisparitiesの最大値を１６の倍数で指定
    ndisparities = 16 * 5
    # ブロック窓のサイズ，最大２１の奇数で指定
    SADWindowSize = 21

    sbm = cv2.StereoBM_create(ndisparities, SADWindowSize)

    # 視差画像を計算
    imgDisparity16S = sbm.compute(imgLeft, imgRight)

    # 視差画像の最小値が０，最大値が２５５になるように線形変換（正規化）して表示
    minVal, maxVal, _, _ = cv2.minMaxLoc(imgDisparity16S)

    imgDisparity8U = (imgDisparity16S * 255 / (maxVal - minVal)).astype(np.uint8)

    cv2.namedWindow(windowDisparity, cv2.WINDOW_NORMAL)
    cv2.imshow(windowDisparity, imgDisparity8U)

    # 視差画像をファイルに保存
    cv2.imwrite("./SBM_sample.jpg", imgDisparity16S)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
