import numpy as np
import cv2

MKSZE = 5
MKWGT = 2


def calc_projection_matrix(op, ip):
    assert len(op) == len(ip)

    B = np.zeros((2 * len(ip), 11), dtype=np.float64)
    C = np.zeros((2 * len(ip),), dtype=np.float64)

    for i, j in zip(range(0, 2 * len(op), 2), range(len(ip))):
        B[i, 0] = op[j][0]
        B[i, 1] = op[j][1]
        B[i, 2] = op[j][2]
        B[i, 3] = 1.0
        B[i, 8] = -ip[j][0] * op[j][0]
        B[i, 9] = -ip[j][0] * op[j][1]
        B[i, 10] = -ip[j][0] * op[j][2]
        C[i] = ip[j][0]

        B[i + 1, 4] = op[j][0]
        B[i + 1, 5] = op[j][1]
        B[i + 1, 6] = op[j][2]
        B[i + 1, 7] = 1.0
        B[i + 1, 8] = -ip[j][1] * op[j][0]
        B[i + 1, 9] = -ip[j][1] * op[j][1]
        B[i + 1, 10] = -ip[j][1] * op[j][2]
        C[i + 1] = ip[j][1]

    # 方程式を解く
    retval, pp = cv2.solve(B, C, flags=cv2.DECOMP_SVD)

    proj_matrix = np.ones((3, 4), np.float64)
    for i in range(11):
        proj_matrix[int(i / 4), int(i % 4)] = pp[i]

    # 必要であれば透視投影変換行列を分解して内部パラメータ，外部パラメータを求める
    # camera_matrix, rot_matrix, tvec, _, _, _, _ = cv2.decomposeProjectionMatrix(proj_matrix)
    return proj_matrix


def main():
    win_src = 'Source'
    cv2.namedWindow(win_src)

    file_name = './calibbox.jpg'
    img = cv2.imread(file_name)

    # ipは画像内のピクセル位置
    # opは3次元座標で対応する座標値
    ip = [
        (467, 206),
        (717, 250),
        (469, 383),
        (217, 294),
        (712, 543),
        (507, 734),
        (282, 582),
    ]

    op = [
        (0, 0, 150),
        (0, 150, 150),
        (150, 150, 150),
        (150, 0, 150),
        (0, 150, 0),
        (150, 150, 0),
        (150, 0, 0),
    ]

    # 対応点を入力した２次元位置にxを表示
    for i in range(0, len(ip)):
        ix = ip[i][0]
        iy = ip[i][1]
        cv2.line(img, (ix - MKSZE, iy - MKSZE), (ix + MKSZE, iy + MKSZE), (0, 0, 255), MKWGT)
        cv2.line(img, (ix - MKSZE, iy + MKSZE), (ix + MKSZE, iy - MKSZE), (0, 0, 255), MKWGT)
    cv2.imshow(win_src, img)

    # 入力された行列を用いて，P行列を計算する
    proj_matrix = calc_projection_matrix(op, ip)

    # 確認のために３次元位置を再投影する，再投影位置には○が表示される
    wx = 150
    wy = 150
    wz = 150
    wpos = np.array([wx, wy, wz, 1.0])

    ipos = proj_matrix @ wpos

    ix = ipos[0] / ipos[2]
    iy = ipos[1] / ipos[2]

    cv2.circle(img, (int(ix), int(iy)), MKSZE, (255, 0, 255), MKWGT)
    cv2.imshow(win_src, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
