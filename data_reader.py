# coding=utf-8
import cv2
import os
import random
import numpy as np

class GOTHIC():
    def __init__(self, path, shuffle=True):
        self.path = path
        self.shuffle = shuffle
        folder_list = sorted(os.listdir(self.path))
        print folder_list
        folder_num = len(folder_list)
        self.cls_num = folder_num
        # print self.cls_num
        self.read_list = []
        for i in range(folder_num):
            one_cls = []
            folder = folder_list[i]
            file_list = os.listdir(self.path+folder)
            file_list = sorted(file_list)
            for file_name in file_list:
                path_one = folder+'/'+file_name
                one_cls.append([path_one,i])
            self.read_list.append(one_cls)

        self.simple_num = 0
        for one_cls in self.read_list:
            self.simple_num += len(one_cls)

    def __getitem__(self, index):


        if self.shuffle:
            cls = random.randint(0,self.cls_num-1)
            num = random.randint(0,len(self.read_list[cls])-1)
        else:
            index = index%self.simple_num
            for k in range(self.cls_num):
                if index>=len(self.read_list[k]):
                    index = index - len(self.read_list[k])
                else:
                    cls = k
                    num = index
                    break


        path_img, tag = self.read_list[cls][num]
        img = cv2.imread(self.path+path_img)
        return img, tag

    def __len__(self):
        return self.simple_num

