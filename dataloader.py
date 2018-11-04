from __future__ import print_function, division
import sys
import os
import torch
import pandas as pd
import numpy as np
import random
import csv
import time
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=' '))
        except ValueError as e:
            raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=' '), self.classes)
        except ValueError as e:
            raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

        # import pdb
        # pdb.set_trace()

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise (ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot, vbox, igno = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot, 'vbox': vbox, 'ignore': igno, 'scale': 1}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):

        # img = skimage.io.imread(self.image_names[image_index])
        img = cv2.imread(self.image_names[image_index])

        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))
        vboxs = np.zeros((0, 5))
        ignores = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations, vboxs, ignores

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            x1v = a['x1v']
            x2v = a['x2v']
            y1v = a['y1v']
            y2v = a['y2v']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2
            if a['class'] == 'ignore':
                annotation[0, 4] = -2
                ignores = np.append(ignores, annotation, axis=0)
            else:
                annotation[0, 4] = self.name_to_label(a['class'])
                annotations = np.append(annotations, annotation, axis=0)

                vbox = np.zeros((1, 5))
                vbox[0, 0] = x1v
                vbox[0, 1] = y1v
                vbox[0, 2] = x2v
                vbox[0, 3] = y2v
                vbox[0, 4] = self.name_to_label(a['class'])
                vboxs = np.append(vboxs, vbox, axis=0)

        return annotations, vboxs, ignores

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            if len(row) == 10:
                try:
                    img_file, x1, y1, x2, y2, x1v, y1v, x2v, y2v, class_name = row[:10]
                except ValueError:
                    raise (ValueError('line {}: format should be img_file,x1,y1,x2,y2,x1v,y1v,x2v,y2v,class_name'.format(line)))

            else:
                img_file, x1, y1, x2, y2, class_name = row[:6]
                x1v, y1v, x2v, y2v, if_occ = 0, 0, 0, 0, 0

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('.', '.', '.', '.', '.'):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
            x1v = self._parse(x1v, int, 'line {}: malformed x1: {{}}'.format(line))
            y1v = self._parse(y1v, int, 'line {}: malformed y1: {{}}'.format(line))
            x2v = self._parse(x2v, int, 'line {}: malformed x2: {{}}'.format(line))
            y2v = self._parse(y2v, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 < x1 and class_name != 'ignore':
                raise ValueError('line {}: x2v ({}) must be higher than x1v ({})'.format(line, x2v, x1v))
            if y2 < y1 and class_name != 'ignore':
                raise ValueError('line {}: y2v ({}) must be higher than y1v ({})'.format(line, y2v, y1v))
            if x2v < x1v and class_name != 'ignore':
                raise ValueError('line {}: x2v ({}) must be higher than x1v ({})'.format(line, x2v, x1v))
            if y2v < y1v and class_name != 'ignore':
                raise ValueError('line {}: y2v ({}) must be higher than y1v ({})'.format(line, y2v, y1v))

            # check if the current class name is correctly present
            if class_name not in classes and class_name != 'ignore':
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'x1v': x1v, 'x2v': x2v, 'y1v': y1v, 'y2v': y2v, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    vbox = [s['vbox'] for s in data]
    igno = [s['ignore'] for s in data]
    scale = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
    if max_num_annots > 0:
        for idx, annot in enumerate(annots):
            # print(annot.shape)
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    vbox_padded = torch.ones((len(vbox), max_num_annots, 5)) * -1
    if max_num_annots > 0:
        for idx, annot in enumerate(vbox):
            if annot.shape[0] > 0:
                vbox_padded[idx, :annot.shape[0], :] = annot
    else:
        vbox_padded = torch.ones((len(annots), 1, 5)) * -1

    max_num_igno = max(ig.shape[0] for ig in igno)
    ig_padded = torch.ones((len(igno), max_num_igno, 5)) * -1
    if max_num_igno > 0:
        for idx, ig in enumerate(igno):
            if ig.shape[0] > 0:
                ig_padded[idx, :ig.shape[0], :] = ig
    else:
        ig_padded = torch.ones((len(igno), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'vbox': vbox_padded, 'ignore': ig_padded, 'scale': scale}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=800, max_side=1400):
        image, annots, vbox, igno, scale = sample['img'], sample['annot'], sample['vbox'], sample['ignore'], sample['scale']

        rows, cols, cns = image.shape

        # scale = min_side / rows

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale

        image = cv2.resize(image, (int(round((cols * scale))), int(round((rows * scale)))))
        # image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))

        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale
        vbox[:, :4] *= scale
        igno[:, :4] *= scale

        return {'img': new_image, 'annot': annots, 'vbox': vbox, 'ignore': igno, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots, vbox, igno, scale = sample['img'], sample['annot'], sample['vbox'], sample['ignore'], sample['scale']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            x1v = vbox[:, 0].copy()
            x2v = vbox[:, 2].copy()
            xv_tmp = x1v.copy()
            vbox[:, 0] = cols - x2v
            vbox[:, 2] = cols - xv_tmp

            y1 = igno[:, 0].copy()
            y2 = igno[:, 2].copy()
            y_tmp = y1.copy()
            igno[:, 0] = cols - y2
            igno[:, 2] = cols - y_tmp

            sample = {'img': image, 'annot': annots, 'vbox': vbox, 'ignore': igno, 'scale': scale}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots, vbox, igno, scale = sample['img'], sample['annot'], sample['vbox'], sample['ignore'], sample['scale']

        image = (image.astype(np.float32) - self.mean) / self.std

        sample = {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'vbox': torch.from_numpy(vbox),
                  'ignore': torch.from_numpy(igno), 'scale': scale}
        return sample


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


