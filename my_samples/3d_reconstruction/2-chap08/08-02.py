import cv2
import sys

import numpy as np
from tangled_up_in_unicode import name


def main():
    # ウィンドウ名やファイル名に関するパラメータ
    win_src = "Source"
    win_und = "Undistorted Image"
    file_cam_param = "cam_param.xml"

    # チェッカーパターンに関する変数とパラメータ
    NUM_IMG = 5
    PAT_SIZE = (10, 7)
    CHESS_SIZE = 24.0

    # キャリブレーションパターンの読み込み
    imgs = []
    for i in range(NUM_IMG):
        file_name = "./calib_img" + f"{i + 1}" + ".jpg"
        imgs.append(cv2.imread(file_name))

    # 3次元空間座標での交点位置の設定
    obj_pos = []
    for i in range(NUM_IMG):
        j_obj_pos = []
        for j in range(PAT_SIZE[0] * PAT_SIZE[1]):
            j_obj_pos.append([
                j % PAT_SIZE[0] * CHESS_SIZE,
                j // PAT_SIZE[0] * CHESS_SIZE,
                0.0
            ])
        j_obj_pos = np.array(j_obj_pos, dtype=np.float32)
        obj_pos.append(j_obj_pos)

    # チェスボード（キャリブレーションパターン）のコーナー検出
    img_pos = []
    for i in range(NUM_IMG):
        print("calib_img" + f"{i + 1}" + ".jpg")
        cv2.imshow(win_src, imgs[i])

        found_chessboard, found_corners = cv2.findChessboardCorners(imgs[i], PAT_SIZE)
        if found_chessboard:
            img_pos.append(
                found_corners
            )
            cv2.drawChessboardCorners(imgs[i], PAT_SIZE, img_pos[i], True)
            cv2.imshow(win_src, imgs[i])
            print(" - success")
            cv2.waitKey(0)
        else:
            print(" - fail")
            cv2.waitKey(0)
            sys.exit(-1)

    # Zhangの手法によるキャリブレーション
    retval, inner_camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pos, img_pos, imgs[0].shape[1::-1], None, None)

    # 回転ベクトルと平行移動ベクトルを4x4の外部パラメータ行列に書き換え(1枚目の外部パラメータ行列のみ出力)
    extr_camera_matrix = np.eye(4, 4, dtype=np.float64)
    R, Jacob = cv2.Rodrigues(rvecs[0])
    extr_camera_matrix[:3, :3] = R
    extr_camera_matrix[:3, 3] = tvecs[0].reshape(-1)

    # xmlファイルへの書き出し
    f = cv2.FileStorage(file_cam_param, flags=cv2.FILE_STORAGE_WRITE)
    f.write(name='extrinsic', val=extr_camera_matrix)
    f.write(name='intrinsic', val=inner_camera_matrix)
    f.write(name='distortion', val=dist_coeffs)
    f.release()

    # 画像の歪み補正
    for i in range(NUM_IMG):
        img_undist = cv2.undistort(imgs[i], inner_camera_matrix, dist_coeffs)
        cv2.imshow(win_src, imgs[i])
        cv2.imshow(win_und, img_undist)

        print(inner_camera_matrix)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
