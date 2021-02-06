import os
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import glob
import random
from PIL import Image
import xml.etree.ElementTree as ET


def get_from_objects(xml_file,item="bbox"):
    obj_list = xml_file.findall("object")
    bounding_boxes = []
    for cls in obj_list:
        class_name = cls.find("name").text
        if item=="label":
            bounding_boxes.append(class_name)
        elif item=="bbox":
            bbox = [int(cls.find("bndbox").find("xmin").text),int(cls.find("bndbox").find("ymin").text),int(cls.find("bndbox").find("xmax").text),int(cls.find("bndbox").find("ymax").text)]
            bounding_boxes.append((class_name,bbox))
        else:
            raise Exception("Please input 'bbox' or 'label'")
    return bounding_boxes


class RemMarketDataset(Dataset):
    """
    Dataset class for Market Dataset
    """
    def __init__(self,root_dir,
                train=True,
                transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ]),
                train_val_split=0.8,
                in_memory=False,
                seed=2021
                ):
        self.train = train
        self.transform = transform
        self.in_memory = in_memory
        self.label_lookup = self._get_unique_cls_labels()
        self.image_data = self._read_data(root_dir)
        self.data_set = self._train_val_split(split=train_val_split,seed=seed)
        print("init")

    def _read_data(self,path):
        """
        This method read all data paths from given path. images and xml files should be in given path with same names
            desired format is: img001.jpg,img001.xml

        :param path: data set path
        :return: all image and target paths as a tuple
        """
        images = sorted(glob.glob(path+os.sep+"*.jpg"))
        target = sorted(glob.glob(path + os.sep + "*.xml"))
        assert len(images)==len(target)
        dataset = [(images[idx],target[idx]) for idx in range(len(images)) ]
        return dataset

    def _get_unique_cls_labels(self):
        """
        Reads all xml files in given path, uses them to create a look up dictionary to assign integer values to labels

        :return: unique class labels to construct a lookup dictionary to assign "integer labels" to "string class names"
        """
        annotations = sorted(glob.glob(path + os.sep + "*.xml"))
        cls_labels = []
        for file in annotations:
            xml_file = ET.parse(file)
            cls_labels+=get_from_objects(xml_file, "label")
        unique_cls_labels = sorted(set(cls_labels))
        return {cls_label: idx for idx, cls_label in enumerate(unique_cls_labels)}

    def _load_dataset(self,dataset):
        """
        :param dataset: image and target paths
        :return: loaded data set if in_memory set to true
        """
        if self.in_memory:
            return [(self._load_image(dataset[idx][0]), self._load_target(dataset[idx][1])) for idx in range(len(dataset))]
        else:
            return dataset

    def _train_val_split(self, split,seed):
        """
        :param split: train-validation split ratio
        :param seed: default value is set to keep train and validation sets unique in relevant data set objects
        :return: return loaded dataset.
            based on "self.train" boolean, return either training or validation data set
        """
        dataset = self.image_data.copy()
        random.Random(seed).shuffle(dataset)
        if self.train:
            data = dataset[:int(len(dataset) * split)]
            return self._load_dataset(data)
        elif not self.train:
            data = dataset[int(len(dataset)*split):]
            return self._load_dataset(data)
        else:
            raise Exception("Training mode should be a Boolean")

    def _load_image(self,path):
        """
        :param path: path of image to be loaded
        :return: read image from folder and transform it
        """
        img = Image.open(path)
        return self.transform(img)

    def _load_target(self, path):
        """
        Reads and loads bounding boxes of objects from given xml file. Labels then assigned to integer indexes

        :param path: path of target to be loaded
        :return: read xml file, parse bounding boxes
        """
        xml_file = ET.parse(path)
        objects = get_from_objects(xml_file,"bbox")
        # TODO convert return type to tensor!!!
        return [(self.label_lookup[obj[0]], obj[1] ) for obj in objects]

    def __len__(self):
        """
        :return: returns length of current data set (training or validation)
        """
        return len(self.data_set)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.data_set[idx]
        else:
            return (
                self._load_image(self.data_set[idx]),
                self._load_target(self.data_set[idx])
            )


path = os.path.join(Path(__file__).parents[2],"data","train")
dataset = RemMarketDataset(path,in_memory=True)

print("done")