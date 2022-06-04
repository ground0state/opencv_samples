import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Particle():
    pos: Iterable  # (x,y) shape (2,)
    vel: Iterable  # (v_x,v_y) shape (2,)
    like: float
    weight: float
    keep: bool

    def __post_init__(self):
        self.pos = np.array(self.pos).reshape(2,)
        self.vel = np.array(self.vel).reshape(2,)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.like == other.like
        else:
            kls = other.__class__.__name__
            raise NotImplementedError(
                f'comparison between {self.__class__.__name__} and {kls} is not supported')

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.like < other.like
        else:
            kls = other.__class__.__name__
            raise NotImplementedError(
                f'comparison between {self.__class__.__name__} and {kls} is not supported')

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)


def main():
    WIN_PF = 'main'
    RESIZE = (640, 360)

    seed = 0
    random_state = np.random.RandomState(seed)

    # 1step更新
    dt = 1

    cap = cv2.VideoCapture('04-06.wmv')
    cv2.namedWindow(WIN_PF, cv2.WINDOW_NORMAL)

    ret, img_src = cap.read()
    img_src = cv2.resize(img_src, RESIZE)
    img_src_rows, img_src_cols = img_src.shape[0], img_src.shape[1]

    # パーティクル初期化 画面全体に一様分布
    # 初期尤度1.0 初期重み0.0
    num = 1000
    particle_list = []
    for i in range(num):
        pt = (
            random_state.randint(0, img_src_cols),
            random_state.randint(0, img_src_rows)
        )
        particle_list.append(
            Particle(pt, (0, 0), 1.0, 0.0, False)
        )

    while True:
        # 予測
        for i_paticle in particle_list:
            i_paticle.pos += i_paticle.vel * dt

        ret, img_src = cap.read()

        if not ret:
            break

        img_src = cv2.resize(img_src, RESIZE)

        img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
        vec_hsv = cv2.split(img_hsv)

        # 色相値、彩度値から尤度を計算、更新
        for i_paticle in particle_list:
            pos_x = i_paticle.pos[0]
            pos_y = i_paticle.pos[1]

            if 0 < pos_x < img_src_cols and 0 < pos_y < img_src_rows:
                # 尤度を計算
                h = vec_hsv[0][pos_y, pos_x]
                s = vec_hsv[1][pos_y, pos_x]
                len_h = abs(70 - h)
                len_s = abs(200 - s)
                like = (len_h / 180) * 0.8 + (len_s / 255) * 0.2
                i_paticle.like = 1 - like
            else:
                i_paticle.like = 0

        # 尤度の昇順にソート
        particle_list.sort()

        # 尤度の高いパーティクルを残し、尤度の低いパーティクルを消す
        thresh_like = 0.9
        # 少なくとも残すパーティクル数
        thresh_keep = len(particle_list) / 100

        for i, i_paticle in enumerate(particle_list):
            if i_paticle.like > thresh_like or i > len(particle_list) - thresh_keep:
                i_paticle.keep = True
            else:
                i_paticle.keep = False
        particle_list = [i_paticle for i_paticle in particle_list if i_paticle.keep]

        # 尤度の高いパーティクルの計数、尤度の合計
        like_sum = sum([i_paticle.like for i_paticle in particle_list])

        # 正規化した重みを計算
        for i_paticle in particle_list:
            i_paticle.weight = i_paticle.like / like_sum

        # リサンプリング
        particle_list_new = []
        for i_paticle in particle_list:
            num_new = int(i_paticle.weight * (num - len(particle_list)))
            for _ in range(num_new):
                r = random_state.normal(loc=0.0, scale=sum(img_src.shape)) * (1 - i_paticle.like)
                ang = random_state.uniform(-np.pi, np.pi)
                pt = np.round(
                    np.array([
                        r * np.cos(ang) + i_paticle.pos[0],
                        r * np.sin(ang) + i_paticle.pos[1]
                    ])
                ).astype(np.int64)
                particle_list_new.append(
                    Particle(pt, pt - i_paticle.pos, i_paticle.like, i_paticle.weight, False)
                )

        # パーティクル更新
        particle_list = particle_list + particle_list_new

        # パーティクル描画
        for i_paticle in particle_list:
            pos_x = i_paticle.pos[0]
            pos_y = i_paticle.pos[1]

            if 0 < pos_x < img_src_cols and 0 < pos_y < img_src_rows:
                img_src = cv2.circle(
                    img=img_src,
                    center=i_paticle.pos,
                    radius=2,
                    color=(0, 0, 255),
                    thickness=-1)

        # 追加パーティクル描画
        for i_paticle in particle_list_new:
            img_src = cv2.circle(
                img=img_src,
                center=i_paticle.pos,
                radius=2,
                color=(255, 0, 0),
                thickness=-1)

        # パーティクルの重心
        center = np.round(
            sum([i_paticle.pos for i_paticle in particle_list]) / len(particle_list)
        ).astype('int')
        cv2.line(img_src,
                 (center[0], 0),
                 (center[0], img_src_rows),
                 (0, 255, 255),
                 3)
        cv2.line(img_src,
                 (0, center[1]),
                 (img_src_cols, center[1]),
                 (0, 255, 255),
                 3)

        cv2.imshow(WIN_PF, img_src)

        if cv2.waitKey(1) == ord('q'):
            break

        # 描画のための遅延
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
