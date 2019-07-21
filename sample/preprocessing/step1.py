import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
import SimpleITK as sitk
from glob import glob
import os.path as P


def load_mhd(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions
    # to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    # Read the origin of the ct_scan, will be used to convert the coordinates
    # from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, spacing


def load_egfr(path):
    # Reads the image using SimpleITK

    filenames = sorted(glob(P.join(path, '*.dcm')))

    scans = []
    spacings = []

    for filename in filenames:
        dcm = pydicom.read_file(filename)

        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)
        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        spacing[0] = float(dcm.SliceThickness)

        scans.append(ct_scan)
        spacings.append(tuple(spacing))

    assert len(set(spacings)) == 1, 'spacings FUCK'

    scans.reverse()
    return np.concatenate(scans), np.array(spacings[0])


def load_scan(path):
    """
    读取一个病人的全部slides
    （已测试 7.17）

    :param path: 病人文件夹所在路径
    :return:
    """
    assert isinstance(path, str)
    assert os.path.isdir(path)

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # z轴递增
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        # TODO 何用？
        sec_num = 2  # 统计有几个切片z轴相同（即为同一张图）
        while slices[0].ImagePositionPatient[2] == \
                slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
        print("进入load_scan的if分支")
        slice_num = int(len(slices) / sec_num)  # 断层影像数量
        slices.sort(key=lambda x: float(x.InstanceNumber))  # 多余操作？
        slices = slices[0:slice_num]  # 截取都是相同z值的slide有啥用？
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] -
            slices[1].ImagePositionPatient[2]
        )
    except AttributeError:
        slice_thickness = np.abs(
            slices[0].SliceLocation -
            slices[1].SliceLocation  # SliceLocation与ImagePositionPatient[2]相同
        )
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    """
    获取一个病人的HU单位的numpy矩阵，及spacing参数（内含片间距，和Pixel Spacing）
    注：Pixel Spacing (0028,0030) specifies the physical distance in the patient
    between the center of each pixel.
    （已测试 7.17）

    :param slices: 属于一个病人的所有slice的对象列表
    :return: 一个病人的np矩阵（D,H,W），和空间信息（SliceThickness, PixelSpacing[0],
    PixelSpacing[1]）
    """
    assert isinstance(slices, list) and len(slices) > 1
    assert isinstance(slices[0], pydicom.dataset.FileDataset)

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), should be possible as values
    # should always be low enough (<32k)
    image = image.astype(np.int16)

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        slope = slices[slice_number].RescaleSlope  # 斜率
        intercept = slices[slice_number].RescaleIntercept  # 截距

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
    return (
        np.array(image, dtype=np.int16),
        np.array([slices[0].SliceThickness] + list(slices[0].PixelSpacing),
                 dtype=np.float32)
    )


def binarize_per_slice(
        image,
        spacing,
        intensity_th=-600.0,  # 肺部HU范围-500~-900
        sigma=1,
        area_th=30,
        eccen_th=0.99,
        bg_patch_size=10
):
    """
    将一个patient的每个slice二值化
    （已测试，7.21）

    :param image: （3D）slice的numpy矩阵
    :param spacing: patient的(SliceThickness,PixelSpacing[0], PixelSpacing[1])
    PixelSpacing: adjacent row spacing (delimiter) adjacent column spacing
    :param intensity_th: HU亮度阈值
    :param sigma: 高斯核标准差
    :param area_th: （实际）面积阈值
    :param eccen_th: 离心率阈值
    :param bg_patch_size: 每slice左上角所选取的patch大小
    :return: 一个patient对应的全部slice的二值化后的3D numpy矩阵
    """
    # TODO 函数可并行
    assert isinstance(image, np.ndarray
                      ) and image.ndim == 3 and image.dtype == np.int16
    assert isinstance(spacing, np.ndarray) and len(spacing) == 3
    assert isinstance(intensity_th, (int, float))
    assert isinstance(sigma, int)
    assert isinstance(area_th, int)
    assert isinstance(eccen_th, float)
    assert isinstance(bg_patch_size, int)
    bw = np.zeros(image.shape, dtype=bool)  # 二值化的窗口binarized window

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(
        -image_size / 2 + 0.5, image_size / 2 - 0.5, image_size
    )
    x, y = np.meshgrid(grid_axis, grid_axis)  # 2个2D坐标网格
    d = (x ** 2 + y ** 2) ** 0.5  # 求每个坐标点距离slice中心的像素距离
    nan_mask = (d < image_size / 2).astype(float)  # 中心为1，呈白色
    # nan_mask[nan_mask == 0] = np.nan  # TODO 不必要 边界赋值nan

    for i in range(image.shape[0]):  # 遍历每一slice
        # Check if corner pixels are identical, if so the slice before Gaussian
        # filtering #TODO 英语注释不合理
        # 如果左上角值一小块相同
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            # step1 扣出内接圆区域
            current_bw = np.multiply(image[i].astype('float32'), nan_mask)
            # step2 对整张slice高斯模糊
            current_bw = scipy.ndimage.filters.gaussian_filter(
                current_bw,
                sigma, truncate=2.0
            )
            # step3 构建HU<阈值的mask（挑出肺部）
            # print('current_bw.dtype =', current_bw.dtype)
            # print('type(intensity_th) =', type(intensity_th))
            # print(current_bw.sum())
            # print(current_bw.max())
            # print(current_bw.min())
            # print("=" * 10)
            current_bw = current_bw < intensity_th
        else:  # 如果左上角值一小块不相同
            # step1 对整张slice高斯模糊（跳过上一步扣内接圆步骤）
            current_bw = scipy.ndimage.filters.gaussian_filter(
                image[i].astype('float32'),
                sigma, truncate=2.0
            )
            # step2 构建HU<阈值的mask（挑出肺部）
            current_bw = current_bw < intensity_th

        # select proper components 选择连通分量
        # plt.imshow(current_bw, 'gray')
        label = measure.label(current_bw)  # 2D矩阵
        properties = measure.regionprops(label, coordinates='xy')
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th \
                    and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.isin(label, list(valid_label))  # in1d更新为isin
        # if i == 80:
        #     print(current_bw.dtype)
        #     plt.imshow(current_bw, 'gray')
        #     plt.show()
        # # current_bw=current_bw.reshape(label.shape)
        bw[i] = current_bw
    return bw


def all_slice_analysis(
        bw,
        spacing,
        cut_num=0,
        vol_limit=(0.68, 8.2),
        area_th=6e3,  # mm^2
        dist_th=62,  # mm
):
    """
    分析一个patient的全部slice

    :param bw: 二值化后的3D矩阵,shape = (D, H, W)
    :param spacing: patient的(SliceThickness,PixelSpacing[0], PixelSpacing[1])
    PixelSpacing: adjacent row spacing (delimiter) adjacent column spacing
    :param cut_num: 需要舍弃的(最上层)slice数量
    :param vol_limit: 体素体积上下限阈值
    :param area_th: 单slice像素面积阈值
    :param dist_th: 连通分量距离volumes中心线的像素距离阈值
    :return:
    """
    assert isinstance(bw, np.ndarray) and bw.ndim == 3 and bw.dtype == bool
    assert isinstance(spacing, np.ndarray) and len(spacing) == 3
    assert isinstance(cut_num, int) and cut_num >= 0
    assert isinstance(vol_limit, tuple) and len(vol_limit) == 2
    assert isinstance(vol_limit[0], float) and vol_limit[0] > 0.
    assert isinstance(vol_limit[1], float) and vol_limit[1] > vol_limit[0]
    assert isinstance(area_th, (int, float)) and area_th > 0
    assert isinstance(dist_th, (int, float)) and dist_th > 0

    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw_backup_0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)  # 3D
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = {
        # 第一张四个角
        label[0, 0, 0], label[0, 0, -1],
        label[0, -1, 0], label[0, -1, -1],
        # 最后一张四个角
        label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1],
        label[-1 - cut_num, -1, 0], label[-1 - cut_num, -1, -1],
        # 第一张中部的上下
        label[0, 0, mid], label[0, -1, mid],
        # 最后一张中部的上下
        label[-1 - cut_num, 0, mid],
        label[-1 - cut_num, -1, mid]
    }
    for l in bg_label:
        label[label == l] = 0

    # 基于体积阈值过滤连通分量
    properties_1 = measure.regionprops(label, coordinates='xy')
    for prop_1 in properties_1:
        if prop_1.area * spacing.prod() < vol_limit[0] * 1e6 \
                or prop_1.area * spacing.prod() > vol_limit[1] * 1e6:  # 超过体积阈值
            label[label == prop_1.label] = 0

    # prepare a distance map for further analysis
    # TODO X/Y轴是否弄混了？
    # x_axis = np.linspace(-label.shape[1] / 2 + 0.5,
    #                      label.shape[1] / 2 - 0.5,
    #                      label.shape[1]
    #                      ) * spacing[1]
    # y_axis = np.linspace(-label.shape[2] / 2 + 0.5,
    #                      label.shape[2] / 2 - 0.5,
    #                      label.shape[2]) * spacing[2]
    # x, y = np.meshgrid(x_axis, y_axis)
    x_axis = np.linspace(-label.shape[2] / 2 + 0.5,  # 按自己思路交换坐标轴修改
                         label.shape[2] / 2 - 0.5,
                         label.shape[2], dtype=np.float32
                         ) * spacing[2]
    y_axis = np.linspace(-label.shape[1] / 2 + 0.5,
                         label.shape[1] / 2 - 0.5,
                         label.shape[1], dtype=np.float32
                         ) * spacing[1]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5  # 2D矩阵
    properties_2 = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all
    # slices
    for prop_2 in properties_2:
        single_vol = np.array(label == prop_2.label)
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):  # 遍历每一张slice
            slice_area[i] = single_vol[i].sum() * spacing[1:3].prod()
            min_distance[i] = (
                    single_vol[i] * d + (1 - single_vol[i]) * d.max()
            ).min()

        if np.average([min_distance[i] for i in range(label.shape[0])
                       if slice_area[i] > area_th]) < dist_th:
            # 若满足要求的slice内，连通区域距离中心线平均距离小于阈值
            valid_label.add(prop_2.label)

    bw = np.isin(label, list(valid_label))  # bool类型 最终处理好的连通区域

    # fill back the parts removed earlier
    if cut_num > 0:  # TODO 不太懂这一部分逻辑
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of
        # their intersection is returned as final mask（有多余计算）
        bw_backup_1 = np.copy(bw)
        # TODO 下一步的意义？？？
        bw_backup_1[-cut_num:] = bw_backup_0[-cut_num:]  # 来自最原始输入
        bw_backup_2 = np.copy(bw)
        bw_backup_2 = scipy.ndimage.binary_dilation(  # 膨胀
            bw_backup_2, iterations=cut_num,  # 重复次数？
        )
        bw3 = bw_backup_1 & bw_backup_2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})  # bw的连通分量的label
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[
                indices[0][0],
                indices[1][0],
                indices[2][0]
            ]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.isin(label3, list(valid_l3))

    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = {
        label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1],
        label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0],
        label[-1, -1, -1]
    }
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2],
                                                   bb[1]:bb[3]] | properties[
                                                       j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[
                                                          bb[0]:bb[2], bb[1]:bb[
                    3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[
            1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False,
                                                             sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False,
                                                             sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


def step1_python(case_path, isEGFR=False):
    assert isinstance(case_path, str)
    assert isinstance(isEGFR, bool)

    if case_path[-4:] == ".mhd":  # TODO 干嘛用的？
        case_pixels, spacing = load_mhd(case_path)
    else:
        if isEGFR:
            case_pixels, spacing = load_egfr(case_path)
        else:
            case = load_scan(case_path)  # 读取一个病人的全部slide
            case_pixels, spacing = get_pixels_hu(case)

    bw = binarize_per_slice(case_pixels, spacing)  # 二值化

    flag = 0  # TODO 什么意义？
    cut_num = 0
    cut_step = 2
    bw_backup = np.copy(bw)  # 原始备份
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw_backup)
        bw, flag = all_slice_analysis(
            bw, spacing, cut_num=cut_num,
            vol_limit=(0.68, 7.5)
        )
        cut_num = cut_num + cut_step

    # bw = fill_hole(bw)
    # bw1, bw2, bw = two_lung_only(bw, spacing)
    # return case_pixels, bw1, bw2, spacing


if __name__ == '__main__':
    INPUT_FOLDER = os.path.expanduser('~/DSB3/stage1/')
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()

    case_pixels, m1, m2, spacing = step1_python(
        os.path.join(INPUT_FOLDER, patients[25])
    )
    # plt.imshow(m1[60])
    # plt.figure()
    # plt.imshow(m2[60])

#     first_patient = load_scan(INPUT_FOLDER + patients[25])
#     first_patient_pixels, spacing = get_pixels_hu(first_patient)
#     plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.show()

#     # Show some slice in the middle
#     h = 80
#     plt.imshow(first_patient_pixels[h], cmap=plt.cm.gray)
#     plt.show()

#     bw = binarize_per_slice(first_patient_pixels, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     flag = 0
#     cut_num = 0
#     while flag == 0:
#         bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num)
#         cut_num = cut_num + 1
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     bw = fill_hole(bw)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     bw1, bw2, bw = two_lung_only(bw, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
