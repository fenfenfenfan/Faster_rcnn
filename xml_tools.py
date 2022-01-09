from lxml import etree

def read_xml(xml_path):
    """
    Args:xml file name or path
    return:Element
    """
    xml = etree.parse(xml_path)
    root = xml.getroot()
    return root


def parse_xml_to_dict(root):
    """
    Args:Element
    return:dict include XML data
    """
    if len(root) == 0:
        return {root.tag:root.text}
    # 递归调用将element对象所有节点转换为字典
    result = {}
    for child in root:
        child_result = parse_xml_to_dict(child)
        if child.tag !="object":
            result[child.tag] = child_result[child.tag]
        else:
            # 将多对象用列表存储
            # if child.tag not in result:
            #     object = []
            # object.append(child_result[child.tag])
            # result[child.tag] = object
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {root.tag:result}


