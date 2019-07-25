"""
测试代码
"""
import os
from sample.preprocessing.step1 import load_scan, get_pixels_hu, \
    binarize_per_slice, all_slice_analysis, fill_hole
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage

INPUT_FOLDER = os.path.expanduser('~/DSB3/stage1/')


def load_scan_test():
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    print("os.path.join(INPUT_FOLDER, patients[25]) =",
          os.path.join(INPUT_FOLDER, patients[25]))

    slices = load_scan(os.path.join(INPUT_FOLDER, patients[25]))
    print(slices)


def get_pixels_hu_test():
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    # print("os.path.join(INPUT_FOLDER, patients[25]) =",
    #       os.path.join(INPUT_FOLDER, patients[25]))

    slices = load_scan(os.path.join(INPUT_FOLDER, patients[25]))
    case_pixels, spacing = get_pixels_hu(slices)
    print(case_pixels)
    print(case_pixels.shape)
    print(case_pixels.dtype)
    print(case_pixels.max())
    print(case_pixels.min())
    print(spacing)
    print(spacing.dtype)
    # plt.hist(
    #     case_pixels.reshape(-1),
    #     bins=100,
    # )
    # plt.title("histogram")
    # plt.show()

    plt.imshow(case_pixels[60, 50:-50, 100:-100], cmap="gray")
    plt.show()
    print(spacing.ndim)
    print(spacing.shape)
    print(len(spacing))


def binarize_per_slice_test():
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    # print("os.path.join(INPUT_FOLDER, patients[25]) =",
    #       os.path.join(INPUT_FOLDER, patients[25]))

    slices = load_scan(os.path.join(INPUT_FOLDER, patients[25]))
    case_pixels, spacing = get_pixels_hu(slices)
    bw = binarize_per_slice(case_pixels, spacing)
    print(type(bw))


def all_slice_analysis_test():
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    # print("os.path.join(INPUT_FOLDER, patients[25]) =",
    #       os.path.join(INPUT_FOLDER, patients[25]))

    slices = load_scan(os.path.join(INPUT_FOLDER, patients[25]))
    case_pixels, spacing = get_pixels_hu(slices)
    bw = binarize_per_slice(case_pixels, spacing)

    flag = 0
    cut_num = 2
    cut_step = 2
    bw, flag = all_slice_analysis(
        bw, spacing, cut_num=cut_num,
        vol_limit=(0.68, 7.5)
    )
    print(bw.shape, bw.dtype)
    print(flag)


def fill_hole_test():
    bw = np.zeros((3, 3, 3), dtype=np.bool)
    bw[1, 1, 1] = True
    bw = fill_hole(bw)
    print(bw)


if __name__ == '__main__':
    # load_scan_test()
    # get_pixels_hu_test()
    # binarize_per_slice_test()
    # all_slice_analysis_test()
    # fill_hole_test()
    np.set_printoptions(precision=3)
    a = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.bool)
    b = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.bool)
    print(a)
    print(b)
    print(a | b)
    print(a & b)
    pass
