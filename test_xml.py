"""
this file is to test the use of reading XML files

"""
from lxml import etree
import os


# 将XML文件解析成字典
def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag:xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag !="object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag:result}


# 从string中读取XML数据
def xml_from_str(xml_name):
    anno_path = "/home/wushaojin/data/VOCdevkit/VOC2012/Annotations"
    xml_path = os.path.join(anno_path, xml_name)

    with open(xml_path) as f:
       xml_str = f.read()
    # print(xml_str)
    xml = etree.fromstring(xml_str)  # fromstring返回根节点,即Element
    print("xml:",xml)
    print(xml.tag)


# 从xml文件中读取XML数据
def xml_from_file(xml_name):
    anno_path = "/home/wushaojin/data/VOCdevkit/VOC2012/Annotations"
    xml_path = os.path.join(anno_path, xml_name)

    xml = etree.parse(xml_path)      # parse直接解析xml文件
    print(xml)
    root = xml.getroot()
    # print(root)
    # print(root.tag)
    # print("tag_length:",len(root))
    # for idx,child in enumerate(root.iter("name")):
    #     print(idx)
    #     print(len(child))
    
    print("element tree")
    data = parse_xml_to_dict(root)
    print(data)


if __name__ == "__main__":
    xml_from_file("2007_000032.xml")
