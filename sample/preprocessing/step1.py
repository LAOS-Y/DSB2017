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
        area_th=6e3,  # mm^2 注意这两个参数用在EGFR是否合适
        dist_th=62,  # mm 注意这两个参数用在EGFR是否合适
):
    """
    分析一个patient的全部slice
    （已测试 7.22）

    :param bw: 二值化后的3D矩阵,shape = (D, H, W)
    :param spacing: patient的(SliceThickness,PixelSpacing[0], PixelSpacing[1])
    PixelSpacing: adjacent row spacing (delimiter) adjacent column spacing
    :param cut_num: 需要舍弃的(最上层)slice数量
    :param vol_limit: 体素体积上下限阈值
    :param area_th: 单slice像素面积阈值
    :param dist_th: 连通分量距离volumes中心线的像素距离阈值
    :return: 经二值化、过滤后的本case矩阵，实际筛选出的连通区域（肺部）数量（若为0则没找到）
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
    bw_backup_0 = None  # 为了消除IDE的错误检查预告
    if cut_num > 0:
        bw_backup_0 = np.copy(bw)
        bw[-cut_num:] = False

    labels = measure.label(bw, connectivity=1)  # 3D (29个连通区域) 4连通

    # remove components access to corners（仅保留属于人体的圆形内组织）
    mid = int(labels.shape[2] / 2)  # W轴中点
    bg_label = {
        # 第一张四个角
        labels[0, 0, 0], labels[0, 0, -1],
        labels[0, -1, 0], labels[0, -1, -1],
        # (cut后)最后一张四个角
        labels[-1 - cut_num, 0, 0], labels[-1 - cut_num, 0, -1],
        labels[-1 - cut_num, -1, 0], labels[-1 - cut_num, -1, -1],
        # 第一张中部的上下
        labels[0, 0, mid], labels[0, -1, mid],
        # 最后一张中部的上下（能够去除圆形区域）
        labels[-1 - cut_num, 0, mid], labels[-1 - cut_num, -1, mid]
    }
    for l in bg_label:
        labels[labels == l] = 0

    # prepare a distance map for further analysis
    # 基于体积阈值过滤连通区域 （基本就可以把肺保留下来了）
    # TODO X/Y轴是否弄混了？
    # x_axis = np.linspace(-label.shape[1] / 2 + 0.5,
    #                      label.shape[1] / 2 - 0.5,
    #                      label.shape[1]
    #                      ) * spacing[1]
    # y_axis = np.linspace(-label.shape[2] / 2 + 0.5,
    #                      label.shape[2] / 2 - 0.5,
    #                      label.shape[2]) * spacing[2]
    x_axis = np.linspace(-labels.shape[2] / 2 + 0.5,  # 按自己思路交换坐标轴修改
                         labels.shape[2] / 2 - 0.5,
                         labels.shape[2], dtype=np.float32) * spacing[2]
    y_axis = np.linspace(-labels.shape[1] / 2 + 0.5,
                         labels.shape[1] / 2 - 0.5,
                         labels.shape[1], dtype=np.float32) * spacing[1]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5  # 2D矩阵 float32
    valid_label = set()
    props_1 = measure.regionprops(labels, coordinates='xy')  # 获取连通区域属性
    for prop_1 in props_1:
        # 如果超过体积不在阈值范围内，置0抹去（不是所需的连通区域）
        if prop_1.area * spacing.prod() < vol_limit[0] * 1e6 \
                or prop_1.area * spacing.prod() > vol_limit[1] * 1e6:
            labels[labels == prop_1.label] = 0
        else:  # 是所需的连通区域
            # select components based on their area and distance to center axis
            # on all slices
            single_vol = np.array(labels == prop_1.label)  # (195, 512, 512)bool

            slice_area = np.zeros(labels.shape[0])
            min_distance = np.zeros(labels.shape[0])

            for i in range(labels.shape[0]):  # 遍历每一张slice
                slice_area[i] = single_vol[i].sum() * spacing[1:3].prod()
                min_distance[i] = (
                        single_vol[i] * d + (1 - single_vol[i]) * d.max()
                ).min()

            # 过滤掉可能的distracting连通区域
            if np.average([min_distance[i] for i in range(labels.shape[0])
                           if slice_area[i] > area_th]) < dist_th:
                # 若满足要求的slice内，连通区域距离中心线平均距离小于阈值
                valid_label.add(prop_1.label)

    # (195, 512, 512) bool
    bw = np.isin(labels, list(valid_label))  # bool类型 最终处理好的连通区域

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw_backup_1 is bw with removed slices, bw_backup_2 is a dilated
        # version of bw, part of their intersection is returned as final mask
        bw_backup_1 = np.copy(bw)
        bw_backup_1[-cut_num:] = bw_backup_0[-cut_num:]  # 来自最原始输入

        bw_backup_2 = np.copy(bw)
        bw_backup_2 = scipy.ndimage.binary_dilation(  # 膨胀
            bw_backup_2, iterations=cut_num,  # 重复次数？
            # 如果cut后肺部刚好贴边，则cut_num次膨胀后，依然贴边，但是为什么这么做？
            # 为了对bw补充完整cut区域，合理～（就是下一行代码）
        )
        bw3 = bw_backup_1 & bw_backup_2  # 这才是最终的bw
        labels = measure.label(bw, connectivity=1)  # 补cut前的，就2个数字，0、1
        labels3 = measure.label(bw3, connectivity=1)  # 补cut后的，同2个数字，0、1
        l_list = list(set(np.unique(labels)) - {0})  # bw的连通分量的label，应该就1

        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(labels == l)  # 选取肺部对应的index
            l3 = labels3[
                indices[0][0],
                indices[1][0],
                indices[2][0]
            ]
            if l3 > 0:  # 在label3中也是器官，这应该是必然的吧？
                valid_l3.add(l3)
        bw = np.isin(labels3, list(valid_l3))

    return bw, len(valid_label)


def fill_hole(bw):
    """
    填充3D矩阵的空洞（去除3D矩阵8个角落可能的噪音）

    :param bw: 经筛选后的二值化肺部区域
    :return: 处理后的3D布尔矩阵
    """
    assert isinstance(bw, np.ndarray) and bw.dtype == bool and bw.ndim == 3

    labels = measure.label(~bw)  # 筛选非肺部空间

    # idendify corner components
    bg_label = {  # 8个角
        labels[0, 0, 0], labels[0, 0, -1],
        labels[0, -1, 0], labels[0, -1, -1],
        labels[-1, 0, 0], labels[-1, 0, -1],
        labels[-1, -1, 0], labels[-1, -1, -1]
    }

    # 为了过滤掉8个角落存在的可能的噪音（可能和数据集相关）
    bw = ~np.isin(labels, list(bg_label))
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    """
    返回分别包含左右肺的2个3D矩阵，和同时包含2个肺的3D矩阵

    :param bw: bool类型3D矩阵
    :param spacing: spacing参数
    :param max_iter: 最大迭代次数
    :param max_ratio: ？？？
    :return: ？？？？
    """
    assert isinstance(bw, np.ndarray) and bw.ndim == 3 and bw.dtype == np.bool
    assert isinstance(spacing, np.ndarray) and len(spacing) == 3
    assert isinstance(max_iter, int) and max_iter > 0
    assert isinstance(max_ratio, float) and max_ratio > 0.

    def extract_main(bw_, cover=0.95):
        """
        仅保留肺部mask
        （基本看懂）

        :param bw_: 单肺3D布尔矩阵
        :param cover: 主要连通区域占slice的比例
        :return: 处理后仅含肺部的mask
        """
        for i in range(bw_.shape[0]):  # 遍历每一个slice
            current_slice_ = bw_[i]  # 2D
            labels_ = measure.label(current_slice_)  # 应该就0、1
            props_ = measure.regionprops(labels_)
            props_.sort(key=lambda x: x.area, reverse=True)  # 大到小（背景、肺）
            area_ = [prop_.area for prop_ in props_]
            count_ = 0
            sum_ = 0

            while sum_ < np.sum(area_) * cover:  # np.sum(area)不就是slice面积么？
                sum_ += area_[count_]
                count_ = count_ + 1

            filter_ = np.zeros(current_slice_.shape, dtype=bool)
            for j in range(count_):  # 仅0
                bb = props_[j].bbox  # 背景bbox
                # bbox: (min_row, min_col, max_row, max_col)
                filter_[bb[0]:bb[2], bb[1]:bb[3]] = \
                    filter_[bb[0]:bb[2], bb[1]:bb[3]] | props_[
                        j].convex_image  # 并集 （和边界外接框同大小的凸包）
            bw_[i] = bw_[i] & filter_  # 交集 为了什么？

        labels_ = measure.label(bw_)
        props_ = measure.regionprops(labels_)
        props_.sort(key=lambda x: x.area, reverse=True)
        bw_ = labels_ == props_[0].label  # 感觉有很多操作多此一举？

        return bw_

    def fill_2d_hole(bw_):
        """
        填补肺内小洞？

        :param bw_: 全肺3D布尔矩阵
        :return:
        """
        for i in range(bw_.shape[0]):
            current_slice_ = bw_[i]
            label_ = measure.label(current_slice_)
            props_ = measure.regionprops(label_)
            for prop_ in props_:  # 应该有3或2个连通区域的属性
                bb_ = prop_.bbox
                current_slice_[bb_[0]:bb_[2], bb_[1]:bb_[3]] = \
                    current_slice_[
                    bb_[0]:bb_[2],
                    bb_[1]:bb_[3]
                    ] | prop_.filled_image
                # filled_image: Binary region image with filled holes which has
                # the same size as bounding box.
            bw_[i] = current_slice_

        return bw_

    found_flag = False
    iter_count = 0
    bw_backup_0 = np.copy(bw)

    while not found_flag and iter_count < max_iter:
        labels = measure.label(
            bw, connectivity=2  # 8连通
        )
        props = measure.regionprops(labels)
        props.sort(key=lambda x: x.area, reverse=True)  # 按递减排列连通区域

        # 如果找到了左右两个分开的肺
        if len(props) > 1 and props[0].area / props[1].area < max_ratio:
            found_flag = True
            bw1 = np.array(labels == props[0].label, dtype=np.bool)  # 更大的肺
            bw2 = np.array(labels == props[1].label, dtype=np.bool)  # 略小的肺
        else:  # 否则侵蚀
            bw = scipy.ndimage.binary_erosion(bw)  # 一次侵蚀一个像素厚度
            iter_count = iter_count + 1

    if found_flag:  # 找到了两个肺独立的左右肺部
        # 距离1号肺部的距离矩阵
        d1 = scipy.ndimage.morphology.distance_transform_edt(
            np.array(bw1 == False, dtype=np.bool),
            sampling=spacing,
        )
        # 距离2号肺部的距离矩阵
        d2 = scipy.ndimage.morphology.distance_transform_edt(
            np.array(bw2 == False, dtype=np.bool),
            sampling=spacing,
        )

        bw1 = bw_backup_0 & (d1 < d2)  # 过滤出1号肺
        bw2 = bw_backup_0 & (d1 > d2)  # 过滤出2号肺

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:  # 没找到两个肺
        bw1 = bw_backup_0
        bw2 = np.zeros(bw.shape, dtype=np.bool)

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


def step1_python(case_path, is_egfr=False):
    """


    :param case_path: （DSB中）一个病人的CT文件全路径
    :param is_egfr: 是否是EGFR数据
    :return:
    """
    assert isinstance(case_path, str)
    assert isinstance(is_egfr, bool)
    assert os.path.isdir(case_path)

    if case_path[-4:] == ".mhd":  # TODO 干嘛用的？
        case_pixels, spacing = load_mhd(case_path)
    else:
        if is_egfr:
            case_pixels, spacing = load_egfr(case_path)
        else:
            case = load_scan(case_path)  # 读取一个病人的全部slide
            case_pixels, spacing = get_pixels_hu(case)

    bw = binarize_per_slice(case_pixels, spacing)  # 二值化

    flag = 0  # 意义：找到了几个疑似肺的连通区域
    cut_num = 0
    cut_step = 2
    bw_backup = np.copy(bw)  # 原始备份
    while flag == 0 and cut_num < bw.shape[0]:  # 没找到，就去掉顶层的2片slice
        bw = np.copy(bw_backup)
        bw, flag = all_slice_analysis(
            bw, spacing, cut_num=cut_num,
            vol_limit=(0.68, 7.5)
        )
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)  # 去除8个角落的可能噪音
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing


if __name__ == '__main__':
    INPUT_FOLDER = os.path.expanduser('~/DSB3/stage1/')
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()

    case_pixels, m1, m2, spacing = step1_python(
        os.path.join(INPUT_FOLDER, patients[0])
    )
    # plt.imshow(m1[60], 'gray')
    # plt.show()

    plt.imshow(m2[60], 'gray')
    plt.show()

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
