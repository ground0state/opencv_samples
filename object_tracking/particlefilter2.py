import time
from dataclasses import dataclass

import cv2
import numpy as np


class Point2d():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return self.__class__(round(self.x * other), round(self.y * other))

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __radd__(self, other):
        return self.__class__(self.x + other, self.y + other)

    @property
    def xy(self):
        pass

    @xy.getter
    def xy(self):
        return (self.x, self.y)


@dataclass
class Particle():
    pos: Point2d
    vel: Point2d
    like: float
    weight: float
    keep: bool

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
        pt = Point2d(
            random_state.randint(0, img_src_cols),
            random_state.randint(0, img_src_rows)
        )
        particle_list.append(
            Particle(pt, Point2d(0, 0), 1.0, 0.0, False)
        )

    while True:
        # 予測
        for i_particle in particle_list:
            i_particle.pos += i_particle.vel * dt

        ret, img_src = cap.read()

        if not ret:
            break

        img_src = cv2.resize(img_src, RESIZE)

        img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
        vec_hsv = cv2.split(img_hsv)

        # 色相値、彩度値から尤度を計算、更新
        for i_particle in particle_list:
            if 0 < i_particle.pos.x < img_src_cols and 0 < i_particle.pos.y < img_src_rows:
                # 尤度を計算
                h = vec_hsv[0][i_particle.pos.y, i_particle.pos.x]
                s = vec_hsv[1][i_particle.pos.y, i_particle.pos.x]
                len_h = abs(70 - h)
                len_s = abs(200 - s)
                like = (len_h / 180) * 0.8 + (len_s / 255) * 0.2
                i_particle.like = 1 - like
            else:
                i_particle.like = 0

        # 尤度の昇順にソート
        particle_list.sort()

        # 尤度の高いパーティクルを残し、尤度の低いパーティクルを消す
        thresh_like = 0.9
        # 少なくとも残すパーティクル数
        thresh_keep = len(particle_list) / 100

        for i, i_particle in enumerate(particle_list):
            if i_particle.like > thresh_like or i > len(particle_list) - thresh_keep:
                i_particle.keep = True
            else:
                i_particle.keep = False
        particle_list = [i_particle for i_particle in particle_list if i_particle.keep]

        # 尤度の高いパーティクルの計数、尤度の合計
        like_sum = sum([i_particle.like for i_particle in particle_list])

        # 正規化した重みを計算
        for i_particle in particle_list:
            i_particle.weight = i_particle.like / like_sum

        # リサンプリング
        particle_list_new = []
        for i_particle in particle_list:
            num_new = int(i_particle.weight * (num - len(particle_list)))
            for _ in range(num_new):
                r = random_state.normal(loc=0.0, scale=sum(img_src.shape)) * (1 - i_particle.like)
                ang = random_state.uniform(-np.pi, np.pi)
                pt = Point2d(
                    r * np.cos(ang) + i_particle.pos.x,
                    r * np.sin(ang) + i_particle.pos.y
                )
                particle_list_new.append(
                    Particle(pt, pt - i_particle.pos, i_particle.like, i_particle.weight, False)
                )

        # パーティクル更新
        particle_list = particle_list + particle_list_new

        # パーティクル描画
        for i_particle in particle_list:
            if 0 < i_particle.pos.x < img_src_cols and 0 < i_particle.pos.y < img_src_rows:
                img_src = cv2.circle(
                    img=img_src,
                    center=i_particle.pos.xy,
                    radius=2,
                    color=(0, 0, 255),
                    thickness=-1)

        # 追加パーティクル描画
        for i_particle in particle_list_new:
            img_src = cv2.circle(
                img=img_src,
                center=i_particle.pos.xy,
                radius=2,
                color=(255, 0, 0),
                thickness=-1)

        # パーティクルの重心
        center = sum([i_particle.pos for i_particle in particle_list]) / len(particle_list)

        cv2.line(img_src,
                 (center.x, 0),
                 (center.x, img_src_rows),
                 (0, 255, 255),
                 3)
        cv2.line(img_src,
                 (0, center.y),
                 (img_src_cols, center.y),
                 (0, 255, 255),
                 3)

        cv2.imshow(WIN_PF, img_src)

        if cv2.waitKey(1) == ord('q'):
            break

        # 描画のための遅延
        time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
