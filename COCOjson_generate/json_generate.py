#!/user/bin/env python
# _*_ coding: utf-8 _*_
# ==================================================
# @File_name: e2ec-main_ACM -> json_generate
# @Software: PyCharm
# @Author: 张福正
# @Time: 2024/4/17 15:01
# ==================================================

import json
import glob
import cv2 as cv
import os


class tococo(object):
    def __init__(self, jpg_paths, label_path, save_path):
        self.images = []
        self.categories = []
        self.annotations = []
        # 返回每张图片的地址
        self.jpgpaths = jpg_paths
        self.save_path = save_path
        self.label_path = label_path
        # 可根据情况设置类别，这里只设置了一类
        self.class_ids = {'pos': 1}
        self.class_id = 1
        self.coco = {}

    def npz_to_coco(self):
        annid = 0
        for num, jpg_path in enumerate(self.jpgpaths):

            imgname = jpg_path.split('\\')[-1].split('.')[0]
            img = cv.imread(jpg_path)
            jsonf = open(self.label_path + imgname + '.json').read()  # 读取json
            labels = json.loads(jsonf)
            h, w = img.shape[:-1]
            self.images.append(self.get_images(imgname, h, w, num))
            for label in labels:
                # self.categories.append(self.get_categories(label['class'], self.class_id))
                px, py, pw, ph = label['x'], label['y'], label['w'], label['h']
                box = [px, py, pw, ph]
                print(box)
                self.annotations.append(self.get_annotations(box, num, annid, label['class']))
                annid = annid + 1

        self.coco["images"] = self.images
        self.categories.append(self.get_categories(label['class'], self.class_id))
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations
        # print(self.coco)

    def get_images(self, filename, height, width, image_id):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        # 文件名加后缀
        image["file_name"] = filename + '.jpg'
        # print(image)
        return image

    def get_categories(self, name, class_id):
        category = {}
        category["supercategory"] = "Positive Cell"
        # id=0
        category['id'] = class_id
        # name=1
        category['name'] = name
        # print(category)
        return category

    def get_annotations(self, box, image_id, ann_id, calss_name):
        annotation = {}
        w, h = box[2], box[3]
        area = w * h
        annotation['segmentation'] = [[]]
        annotation['iscrowd'] = 0
        # 第几张图像，从0开始
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = float(area)
        # category_id=0
        annotation['category_id'] = self.class_ids[calss_name]
        # 第几个标注，从0开始
        annotation['id'] = ann_id
        # print(annotation)
        return annotation

    def save_json(self):
        self.npz_to_coco()
        label_dic = self.coco
        # print(label_dic)
        instances_train2017 = json.dumps(label_dic)
        # 可改为instances_train2017.json
        f = open(os.path.join(save_path + '\instances_train2017.json'), 'w')
        f.write(instances_train2017)
        f.close()


# 可改为train2017，要对应上面的
jpg_paths = glob.glob('L:\Deeplearning\e2ec-main_ACM\Mydataset\\train2017\*.jpg')
# 现有的标注文件地址
label_path = r'L:\Deeplearning\e2ec-main_ACM\Mydataset\annotations_ori\\'
# 保存地址
save_path = r'L:\Deeplearning\e2ec-main_ACM\Mydataset\annotations'
c = tococo(jpg_paths, label_path, save_path)
c.save_json()

