import os
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import glob
import random
from PIL import Image
import xml.etree.ElementTree as ET
import torch


class MarketDataset(Dataset):
    """
    Dataset class for Market Dataset
    """
    def __init__(self,root_dir,
                train=True,
                transform=transforms.Compose([
                    transforms.Resize((300, 300)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
                train_val_split=0.8,
                in_memory=False,
                seed=2021
                ):
        self.train = train
        self.transform = transform
        self.in_memory = in_memory
        self.label_lookup = self._get_unique_cls_labels(root_dir)
        self.image_data = self._read_data(root_dir)
        self.data_set = self._train_val_split(split=train_val_split,seed=seed)

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

    def _get_unique_cls_labels(self,path):
        """
        Reads all xml files in given path, uses them to create a look up dictionary to assign integer values to labels

        :return: unique class labels to construct a lookup dictionary to assign "integer labels" to "string class names"
        """
        # TODO add background class
        annotations = sorted(glob.glob(path + os.sep + "*.xml"))
        cls_labels = []
        for file in annotations:
            xml_file = ET.parse(file)
            cls_labels+=self._parse_xml_file(xml_file)["labels"]
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

    def _parse_xml_file(self,xml_file):

        img_width = int(xml_file.find("size").find("width").text)
        img_height = int(xml_file.find("size").find("height").text)
        resize_factor = torch.FloatTensor([img_width,img_height,img_width,img_height]).unsqueeze(0)

        obj_list = xml_file.findall("object")
        classes = []
        bounding_boxes = []
        for cls in obj_list:
            class_name = cls.find("name").text.lower()
            classes.append(class_name)
            bbox = [int(cls.find("bndbox").find("xmin").text), int(cls.find("bndbox").find("ymin").text),
                    int(cls.find("bndbox").find("xmax").text), int(cls.find("bndbox").find("ymax").text)]
            bounding_boxes.append(bbox)

        b_boxes = torch.FloatTensor(bounding_boxes)/resize_factor

        return {"labels": classes, "boxes": b_boxes}

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
        content = self._parse_xml_file(xml_file)
        classes = content["labels"]
        b_boxes = content["boxes"]

        coded_labels = [self.label_lookup[cls] for cls in classes]

        return {"labels":torch.FloatTensor(coded_labels),"boxes": b_boxes}

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
                self._load_image(self.data_set[idx][0]),
                self._load_target(self.data_set[idx][1])
            )
