# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/SteelShapeNet', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}  # {'K': '1','KT': '2'}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))  # {'K': 0,'KT': 1}

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split("/")[2]) for d in json.load(f)])  # {'1001','1002',...,'2064'}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # {'1065','1066',...,'2080'}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_view_file_list.json'), 'r') as f:
            view_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # {'1065'}
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []  # 存储每个txt文件的具体目录地址
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))  # ['1001.txt', '1002.txt', ..., '1080.txt']
            # print(fns[0][0:-4])
            if split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]  # ['1001', '1002', ..., '1080']
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            elif split == "view":
                fns = [fn for fn in fns if fn[0:-4] in view_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'K': [0, 1, 2], 'KT': [3, 4, 5, 6]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


# 测试部分
if __name__ == "__main__":
    # 创建数据集实例
    dataset = PartNormalDataset(
        root="./data/SteelShapeNet",
        npoints=2500,
        split="train",
        class_choice=None,
        normal_channel=False,
    )

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # 测试从dataloader中载入数据
    print("The number of training data is: %d" % len(dataset))
    for i, data in enumerate(dataloader, 0):
        point_set, cls, seg = data
        print(f"Batch {i+1}:")
        print(
            f"Point set shape: {point_set.shape}"
        )  # 输出点云数据的形状 (batch_size, npoints, 3或6)
        print(f"Class shape: {cls.shape}")  # 输出类别标签的形状 (batch_size,)
        print(
            f"Segmentation shape: {seg.shape}"
        )  # 输出分割标签的形状 (batch_size, npoints)
        print(f"Class: {cls}")  # 输出类别值
        print(f"Segmentation: {seg}")  # 输出分割标签
        if i == 0:  # 只测试一个批次
            break
