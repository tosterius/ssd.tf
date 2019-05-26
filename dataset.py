import os
from collections import namedtuple
import xml.etree.ElementTree as ET

LabeledObject = namedtuple('LabeledObject', ['label', 'xc', 'yc', 'w', 'h'])
LabeledImage = namedtuple('LabeledImage', ['filepath', 'size', 'objects'])


class VocDataset:
    def __init__(self, root_directory=None):
        self.data = []
        self.label_map = {'background': 0,
                          'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                          'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
                          'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                          'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

        if root_directory is not None:
            self.init(root_directory)

    def init(self, root_directory):
        annotations_root = os.path.join(root_directory, 'Annotations')
        images_root = os.path.join(root_directory, 'Images')
        annotations_files = os.listdir(annotations_root)

        for filename in annotations_files:
            xmlpath = os.path.join(annotations_root, filename)
            try:
                self.__parse_xml(images_root, xmlpath)
            except Exception as e:
                print(str(e))

    def __parse_xml(self, images_root, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()

        filename = root.find('filename').text
        filepath = os.path.join(images_root, filename)
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        labeled_image = LabeledImage(filepath, (img_h, img_w), [])
        for o in root.iter('object'):
            label = self.label_map[o.find('name').text]
            bbox = o.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xc = (xmin + xmax) / 2.0 / img_w
            yc = (ymin + ymax) / 2.0 / img_h
            w = float(xmax - xmin) / img_w
            h = float(ymax - ymin) / img_h

            labeled_image.objects.append(LabeledObject(label=label, xc=xc, yc=yc, w=w, h=h))

        self.data.append(labeled_image)

    def extend(self, other):
        if other.label_map != self.label_map:
            raise RuntimeError("Datasets must have identical map")
        self.data += other.data
        return self


if __name__ == '__main__':
    ds1 = VocDataset()
    ds1 = ds1.extend(VocDataset('/home/arthur/Workspace/projects/github/ssd.tf/VOC2007'))
    ds1 = ds1.extend(VocDataset('/home/arthur/Workspace/projects/github/ssd.tf/VOC2008'))
    pass