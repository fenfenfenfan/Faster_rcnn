import random
from torchvision.transforms import functional as F
# from torchvision.transforms import RandomHorizontalFlip
# from torchvision.transforms import ToTensor
# from torchvision.transforms import Compose


# 以下都是参照pytorch官方源码自定义自己的transforms
# 因为我们在自定义数据集上进行transforms操作时，首先输入变量有两个img、target，其次在水平翻转时需要同时翻转图像和bbox
# 使用官方内置函数是无法做到的
class Compose(object):
    """组合多个transform函数"""
    # transforms输入的是一个list
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:          # random()方法随机在[0,1)范围内返回一个实数 https://www.runoob.com/python/func-number-random.html
            height, width = image.shape[-2:]     # image已经是一个tensor类型，它将图片信息转换为一个多维矩阵，矩阵的高宽即代表图片高宽，直接返回shape的最后两维大小
            image = image.flip(-1)               # 水平翻转图片 https://blog.csdn.net/Ocean_waver/article/details/113814671
            bbox = target["bboxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["bboxes"] = bbox
        return image, target
