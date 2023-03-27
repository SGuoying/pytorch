# modified from https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py

import os, csv
from PIL import Image
from xml.etree import ElementTree as ET

import torch
from torch.utils.data import Dataset

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class TinyImageNet(VisionDataset):
    dataset_name = 'ILSVRC'
    txt_file = 'imagenet-object-localization-challenge'
    raw_file_name = f'{dataset_name}.zip'
    download_url = 'https://image-net.org/challenges/LSVRC/2012/2012-downloads.php'
    md5 = '1d675b47d978889d74fa0da5fadfb00e'  #'29b22e2961454d5413ddabcf34fc5622'   #val 

    def __init__(self, root='.data', split='LOC_', transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform)

        self.root = os.path.abspath(root)
        self.dataset_path = os.path.join(self.root, self.dataset_name)
        self.txt_path = os.path.join(self.root, self.txt_file)
        self.loader = default_loader
        self.split = verify_str_arg(split, ('train_solution', 'val_solution'))

        raw_file_path = os.path.join(self.root, self.raw_file_name)
        if check_integrity(raw_file_path, self.md5) is True:
            print(f'{self.raw_file_name}already downloaded and verified.')
            if not os.path.exists(self.dataset_path):
                print('Extracting...')
                extract_archive(raw_file_path)
        elif os.path.exists(self.dataset_path):
            pass
        elif download is True:
            print('Downloading...')
            download_url(self.download_url, root=self.root, filename=self.raw_file_name)
            print('Extracting...')
            extract_archive(raw_file_path)
        else:
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

        image_to_class_file_path = os.path.join(self.txt_path, f'{self.split}.csv')
        if os.path.exists(image_to_class_file_path):
            self.data = read_from_csv(image_to_class_file_path)
        else:
            classes_file_path = os.path.join(self.txt_path, 'LOC_synset_mapping.txt')
            _, class_to_idx = find_classes(classes_file_path)

            self.data = make_dataset(self.dataset_path, self.split, class_to_idx)
            try:
                write_to_csv(image_to_class_file_path, self.data)
            except Exception:
                print('can not write to csv file.')

    def __getitem__(self, index: int):
        image_path, target = self.data[index]
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.data)



    dataset_name = '/kaggle/input/imagenet-object-localization-challenge'
    def __init__(self, root='', split='', transform = None, ):
        super(SelectedImagenet, self).__init__(root, transform=transform)
        self.root = root
        self.split = split
        self.transform = transform

        # image_to_class_file_path = os.path.join(self.dataset_name, f'LOC_{self.split}_solution.csv')
        image_to_class_file_path = os.path.join(self.dataset_name, f'{self.split}_data.csv')
        if os.path.exists(image_to_class_file_path):
            self.data = read_from_csv(image_to_class_file_path)
        else:
            classes_file_path = os.path.join(self.dataset_name, 'LOC_synset_mapping.txt')
            _, class_to_idx = find_classes(classes_file_path)

            self.data = make_dataset(self.dataset_name, self.split, class_to_idx)
            try:
                write_to_csv(image_to_class_file_path, self.data)
            except Exception:
                print('can not write to csv file.')

    def __getitem__(self, index: int):
        image_path, target = self.data[index]
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.data)
	

class ImageNetDetection(VisionDataset):
    def __init__(self, root, split='train', transform = None):
        super(ImageNetDetection, self).__init__(root, transform=transform)
        self.root = root
        self.split = split
        self.transform = transform
        self.image_names = os.listdir(os.path.join(self.root, 'Data', 'CLS-LOC', self.split))

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root, 'Data', 'CLS-LOC', self.split, img_name)

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        annotation_path = os.path.join(self.root, 'Annotations', 'CLS-LOC', self.split, img_name.replace(".JPEG", ".xml"))
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.float64)
        
        if self.transform:
            img, target = self.transform(img, target)
        
        return img, target

def find_classes(classes_file_path):
    with open(classes_file_path) as f:
        classes = list(map(lambda s: s.strip(), f.readlines()))

    classes.sort()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    return classes, class_to_idx


def make_dataset(dataset_path, split, class_to_idx):
    train = 'ILSVRC/Data/CLS-LOC'
    images = []
    splitted_dataset_path = os.path.join(dataset_path, train, split)

    if split == 'train':
        for class_name in sorted(os.listdir(splitted_dataset_path)):
            class_path = os.path.join(splitted_dataset_path, class_name)
            if os.path.isdir(class_path):
                class_images_path = os.path.join(class_path)
                for image_name in sorted(os.listdir(class_images_path)):
                    image_path = os.path.join(class_images_path, image_name)
                    item = (image_path, class_to_idx[class_name])
                    images.append(item)
    elif split == 'val':
        images_path = os.path.join(splitted_dataset_path, 'images')
        images_annotations = os.path.join(splitted_dataset_path, 'val_annotations.txt')
        with open(images_annotations) as f:
            meta_info = map(lambda s: s.split('\t'), f.readlines())

        image_to_class = {line[0]: line[1] for line in meta_info}
        
        for image_name in sorted(os.listdir(images_path)):
            image_path = os.path.join(images_path, image_name)
            item = (image_path, class_to_idx[image_to_class[image_name]])
            images.append(item)
    else:
        raise RuntimeError("split other than train and val has not been implemented.")

    return images


def write_to_csv(file: str, image_to_class: list):
    with open(file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(image_to_class)

def read_from_csv(file: str):
    with open(file, "r") as f:
        reader = csv.reader(f)
        data = [[row[0], int(row[1])] for row in reader]
    return data


class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, root: str, num_workers: int=2, pin_memory: bool=True, persistent_workers: bool=True, train_transforms=None, val_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_transforms, self.val_transforms = train_transforms, val_transforms
        self.setup()

    def setup(self, stage=None):
        # self.train_data = TinyImageNet(root=self.root, split='train', transform=self.train_transforms)
        # self.val_data = TinyImageNet(root=self.root, split='val', transform=self.val_transforms)
        self.train_data = ImageNetDetection(root=self.root, split='train', transform=self.train_transforms)
        self.val_data = ImageNetDetection(root=self.root, split='val', transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )