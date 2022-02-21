import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from xml_tools import read_xml, parse_xml_to_dict


class VOCDataset(Dataset):
    """
    build VOC2007、VOC2012 Dataset
    """

    def __init__(self,
                 root_path,
                 data_name,
                 transforms=None,
                 txt_name="train.txt"):
        """
        args:
            root_path->the root_path where the file "VOCdevkit" lies in
            data_name->the data which you choose to use, VOC2007 or VOC2012
        function:
            init dataset
            get files path
            get class data from json data
        """
        self.voc_path = os.path.join(root_path, "VOCdevkit", data_name)
        self.txt_path = os.path.join(self.voc_path, "ImageSets", "Main",
                                     txt_name)
        self.JPG_path = os.path.join(self.voc_path, "JPEGImages")
        self.anno_path = os.path.join(self.voc_path, "Annotations")

        # 判断train.txt文件是否存在
        assert os.path.exists(
            self.txt_path), f"path:{self.txt_path} does not exist"
        with open(self.txt_path, "r") as read:
            self.xml_list = [
                os.path.join(self.anno_path,
                             line.strip() + ".xml")
                for line in read.readlines() if len(line.strip()) > 0
            ]

        # 判断xml文件是否存在
        assert len(self.xml_list) > 0, "nothing in {}".format(self.txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found {}".format(xml_path)

        # 读取jason文件转换为字典
        json_file = "./pascal_voc_classes.json"
        assert os.path.exists(json_file), "not found {}".format(json_file)
        with open(json_file, "r") as j:
            self.class_dict = json.load(j)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        """
        read .xml file
        transform xml to dict
        get images
        get bboxes like [xmin,ymin,xmax,ymax]
        get labels
        get iscrowd which represents obeject whether is difficult to be detected or not   
        transform type of data to tensor
        creat data_dict to save above data
        transforms
        """
        # 解析XML得到字典
        xml_path = self.xml_list[idx]
        xml = read_xml(xml_path)
        data = parse_xml_to_dict(xml)["annotation"]

        # 获得JPEG图片
        img_name = data["filename"]
        img_path = os.path.join(self.JPG_path, img_name)
        assert os.path.exists(img_path), f"{img_path} not found image"
        img = Image.open(img_path)
        if img.format != "JPEG":
            raise ValueError(f"Image '{img_path}' not found")

        # 获得bboxes、labels、iscrowed参数
        # labels需要转换为对应的标签编号
        bboxes = []
        labels = []
        iscrowed = []
        assert "object" in data, f"the {xml_path} does not have object information"
        for obj in data["object"]:
            # 取出的是字符，需要转换为float
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            # 判断是否存在长宽为0 bbox continue终止此次循环
            if xmin >= xmax or ymin >= ymax:
                print(f"Warning: path {xml_path} exit bbox w/h <=0")
                continue
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowed.append(int(obj["difficult"]))
            else:
                iscrowed.append(0)

        # 将data转换为tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowed = torch.as_tensor(iscrowed, dtype=torch.int64)
        img_id = torch.tensor([idx])
        # 获得bbox面积
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

        # 将所有tensor存入字典
        data_dict = {}
        data_dict["bboxes"] = bboxes
        data_dict["labels"] = labels
        data_dict["iscrowed"] = iscrowed
        data_dict["img_id"] = img_id
        data_dict["area"] = area

        # transforms
        if self.transforms is not None:
            img, data_dict = self.transforms(img, data_dict)

        return img, data_dict

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
