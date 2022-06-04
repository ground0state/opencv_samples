# %%
"""
notebookとして実行すること。

1. hで抜き出したい部位の色を決める
2. 小さいsは白、小さいvは黒を含むので、sとvの下限を上げていって光の反射や影を除去する。
"""


import cv2
import numpy as np
from ipywidgets import interact, fixed, IntRangeSlider
import IPython.display as display


def imshow(img):
    _, ret = cv2.imencode('.jpg', img)
    im = display.Image(data=ret)
    display.display(im)


def f(hsv, h_range, s_range, v_range):
    hsv_min = np.array([h_range[0], s_range[0], v_range[0]])
    hsv_max = np.array([h_range[1], s_range[1], v_range[1]])
    img_bin = cv2.inRange(hsv, hsv_min, hsv_max)
    imshow(img_bin)


def create_slider():
    initial_range = {
        'h': [0, 255],
        's': [0, 255],
        'v': [50, 255]
    }
    argument = {}
    for k, v in initial_range.items():
        slider = IntRangeSlider(
            value=v, min=0, max=255, step=1, description=f'{k}:'
        )
        slider.style.handle_color = 'lightblue'
        argument[f'{k}_range'] = slider
    return argument


def read_img(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv


# %%
_, hsv = read_img("thum.png")
argument = create_slider()

interact(
    f,
    hsv=fixed(hsv),
    **argument)

# %%
