import cv2
import numpy as np


def main():

    # カメラの内部パラメータに関する変数（キャリブレーションした値を入れる）
    # f = 1094.0  # 焦点距離
    # cx = 491.0  # 画像中心(x) pixel
    # cy = 368.0  # 画像中心(y) pixel

    f = 1100.0  # 焦点距離
    cx = 300.0  # 画像中心(x) pixel
    cy = 500.0  # 画像中心(y) pixel

    # 入力画像に関する変数
    # 入力画像の数
    # NUM_IMG = 5
    NUM_IMG = 3

    # 入力ファイル名のセット
    image_files = []
    for i in range(NUM_IMG):
        # image_files.append(f"./09-06-{i + 1}.jpg")
        image_files.append(f"./resized_IMG_{2889+i}.jpg")

    # 内部パラメータ行列の生成
    K = np.array([
        [f, 0, cx],
        [0, f, cy, ],
        [0, 0, 1]
    ], dtype=np.float32)

    # SfMモジュールを用いた複数の画像データからの３次元再構成（点群が計算される）s
    is_projective = True
    Rs_est, ts_est, K, points3d_estimated = cv2.sfm.reconstruct(
        images=image_files, K=K, is_projective=is_projective)

    print('shape---')
    print('Rs_est:', len(Rs_est), Rs_est[0].shape)  # num_images tuple of array(3, 3)
    print('ts_est:', len(ts_est), ts_est[0].shape)  # num_images tuple of array(3, 1)
    print('K:', K.shape)  # tuple of array(3, 3)
    print('points3d_estimated:', len(points3d_estimated), points3d_estimated[0].shape)  # num_point tuple of array(3, 1)
    print(K)

    # 結果の表示(Vizを使用する）
    window = cv2.viz.Viz3d("Coordinate Frame")
    window.setWindowSize(window_size=(1280, 720))
    # 指定しないと背景は黒
    window.setBackgroundColor(color=cv2.viz.Color().black())

    # 推定された３次元位置をセット
    point_cloud_est = []
    for i in range(len(points3d_estimated)):
        point_cloud_est.append(points3d_estimated[i].T)
    point_cloud_est = np.array(point_cloud_est)

    cloud_widget = cv2.viz.WCloud(point_cloud_est, cv2.viz.Color.white())
    window.showWidget("point_cloud", cloud_widget)

    # カメラ位置のセット
    path = []
    for i in range(len(Rs_est)):
        camera_matrix = cv2.viz.Affine3d(Rs_est[i], ts_est[i]).mat().reshape(1, 16)
        path.append(camera_matrix)
    path = np.array(path)

    trajectory_widget = cv2.viz.WTrajectory(
        path=path,
        display_mode=cv2.viz.PyWTrajectory_BOTH,
        scale=0.2,
        color=cv2.viz.Color.green()
    )
    window.showWidget("cameras_frames_and_lines", trajectory_widget)

    frustums_widget = cv2.viz.WTrajectoryFrustums(path=path, K=K, scale=0.1, color=cv2.viz.Color.yellow())
    window.showWidget("cameras_frustums", frustums_widget)

    pose = cv2.viz.Affine3d(Rs_est[0], ts_est[0])
    window.setViewerPose(pose)

    window.spin()


if __name__ == "__main__":
    main()
