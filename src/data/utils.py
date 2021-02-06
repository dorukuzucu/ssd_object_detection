import glob
import os
from pathlib import Path
import shutil


def split_dataset(path=None,train_test_split=0.85):
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


