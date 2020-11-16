import xml.etree.ElementTree as et
import cv2 as cv
import glob
import numpy as np
import torch
import torch.utils.data as data

LUNA_CLASSES = {
    'nodule' : 0
}

class GroundTruth:
    def __init__(self, filename):
        root = et.parse(filename).getroot()
        size = root.find('size')
        self.width = int(size.find('width').text)
        self.height = int(size.find('height').text)
        self.objects = []
        for obj in root.findall('object'):
            self.objects.append({
                'name': obj.find('name').text,
                'xmin': int(obj.find('bndbox').find('xmin').text),
                'ymin': int(obj.find('bndbox').find('ymin').text),
                'xmax': int(obj.find('bndbox').find('xmax').text),
                'ymax': int(obj.find('bndbox').find('ymax').text),
            })
            
class LUNADataset(data.Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transforms = transform
        self.imgs = sorted(glob.glob(f'{data_root}/*.jpeg'))
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        img = cv.imread(self.imgs[index])
        gt = GroundTruth(self.imgs[index].replace('jpeg','xml'))
        target = np.array([[obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], LUNA_CLASSES[obj['name']]] for obj in gt.objects])

        if self.transform is not None:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)] # to rgb
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def __len__(self):
        return len(self.imgs)