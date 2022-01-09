import os
import random


def main():
    # 固定随机种子
    random.seed(5)

    # 获得文件目录
    root = os.getcwd()
    file_path = os.path.join(root,"VOCdevkit","VOC2007","Annotations")
    assert os.path.exists(file_path),f"path:{file_path} does not exit!"
    
    file_name = sorted([file.split(".")[0]for file in os.listdir(file_path)])
    file_num = len(file_name)

    # 验证集比例
    val_rate = 0.2
    val_idx = random.sample(range(0,file_num),int(file_num * val_rate))

    # 获得验证集和训练集数据名
    val_file = []
    train_file = []
    for idx,file in enumerate(file_name):
        if idx in val_idx:
            val_file.append(file)
        else:
            train_file.append(file)

    # 创建文件，写入训练集和测试集数据
    try:
        val_f = open("val.txt","x")
        train_f = open("train.txt","x")
        val_f.write("\n".join(val_file))
        train_f.write("\n".join(train_file))
    except FileExistError as e:
        print(e)
        exit(1)  # 调用exit函数，终止程序


if __name__=="__main__":
        main()
        