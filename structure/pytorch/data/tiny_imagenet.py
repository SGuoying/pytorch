# modified from https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py

import os, csv

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg


class TinyImageNet(VisionDataset):
    dataset_name = 'tiny-imagenet-200'
    raw_file_name = f'{dataset_name}.zip'
    download_url = 'http://cs231n.standord.edu/tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root='.data', split='train', transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform)

        self.root = os.path.abspath(root)
        self.dataset_path = os.path.join(self.root, self.dataset_name)
        self.loader = default_loader
        self.split = verify_str_arg(split, 'split', ('train', 'val'))

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

        image_to_class_file_path = os.path.join(self.dataset_path, f'{self.split}_data.csv')
        if os.path.exists(image_to_class_file_path):
            self.data = read_from_csv(image_to_class_file_path)
        else:
            classes_file_path = os.path.join(self.dataset_path, 'wnids.txt')
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


def find_classes(classes_file_path):
    with open(classes_file_path) as f:
        classes = list(map(lambda s: s.strip(), f.readlines()))

    classes.sort()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    return classes, class_to_idx


def make_dataset(dataset_path, split, class_to_idx):
    images = []
    splitted_dataset_path = os.path.join(dataset_path, split)

    if split == 'train':
        for class_name in sorted(os.listdir(splitted_dataset_path)):
            class_path = os.path.join(splitted_dataset_path, class_name)
            if os.path.isdir(class_path):
                class_images_path = os.path.join(class_path, 'images')
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