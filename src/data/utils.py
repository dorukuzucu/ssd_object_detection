import os
import glob
import shutil

from pathlib import Path
from torch.utils.data.dataloader import default_collate


def split_dataset(path=None,train_test_split=0.85):
    """
    :param path: Dataset path for images and annotations.
    :param train_test_split: Split ratio for training and test sets
    """
    data_path = os.path.join(Path(__file__).parents[2],"data")
    if path is None:
        path = os.path.join(data_path,"data")+os.sep
    train_path = os.path.join(data_path,"train")
    test_path = os.path.join(data_path, "test")

    data_list = sorted(glob.glob(path+"*"))
    train = data_list[:int(len(data_list)*train_test_split)]
    test = data_list[int(len(data_list)*train_test_split):]

    for data in train:
        shutil.copy2(data,train_path)
    for data in test:
        shutil.copy2(data,test_path)
    print("Data is split into Train and Test sets. Ratio is {}".format(train_test_split))

def collate_fn(batch):
    """
    Since our data size may vary(ie number of objects in an image may change), Pytorch can not create a batch automatically.
    collate_fn function allows pytorch to generate batches from singular inputs
    :param batch: input batch as a list
    :return: A batch from dataset
    """
    items = list(zip(*batch))

    images = default_collate([i for i in items[0]])
    b_boxes = [i["boxes"] for i in items[1]]
    labels = [i["labels"] for i in items[1]]

    return images, {"boxes":b_boxes,"labels":labels}