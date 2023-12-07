import pickle
from typing import Any
import torch
from torch.utils.data import  Dataset, DataLoader
import pandas as pd
import random

# train_dataset_path = '../data/train'
# templates_file = '../data/templates/BGL.log_templates_new.csv'

class LogDataset(Dataset):
    def __init__(self, 
                 seq_dict, 
                 label,
                 seq_keys,
                 max_len=512,
                 replace_rate=0.0,
                 replace_cnt=1,
                 ) -> None:
        super().__init__()
        self.max_len = max_len
        self.replace_rate = replace_rate
        self.replace_cnt = replace_cnt
        self.seq_dict = seq_dict
        self.seq_keys = seq_keys
        self.label_df = label
        self.same_cnt = 0
        self.pass_replace_cnt = 0
        self.same_cnt_all = 0
        self.pass_replace_cnt_all = 0
        if isinstance(label, dict):
            self.label_dict = label
        else:
            self.label_dict = {}
            for row in label.itertuples():
                blk_id = row.BlockId
                label = [1, 0] if (row.Label=='Normal') else [0, 1]
                self.label_dict[blk_id] = label
        # print("{}/{} abnormal label.".format(sum([i[1] for i in self.label_dict.values()]), len(self.label_dict.values())))
        # self.seq_keys = list(self.seq_dict.keys())
        self.random_replace_item = {}

    def __getitem__(self, index):
        try:
            key = self.seq_keys[index]
            # key = self.seq_keys[seq_idx]
            item:list = self.seq_dict[key]
            if self.replace_rate>0:
                item, replace_label = self.__replace(key, item)
            else:
                replace_label = torch.FloatTensor([1, 0])
            group_label = [1, 0] 
            for it in item:
                if it['group_label']==[0, 1]:
                    group_label = [0, 1]
                    break
            return item, torch.FloatTensor(self.label_dict[key]), \
                replace_label, torch.FloatTensor(group_label)
        except Exception as e:
            print(f"index: {index}")
            raise e
        
    def __replace(self, key, item:list):
        replace_label = []
        # 如果 random_replace_flag 中还未保存且随机到了需要替换，或者保存的值为True，则替换
        # if key in self.random_replace_item.keys():
        #     # return self.random_replace_item[key]
        #     replace_flag = ((torch.argmax(self.random_replace_item[key][1]))==torch.argmax(torch.FloatTensor([0, 1])))
        # else:
        #     replace_flag = random.random()<self.replace_rate

        replace_flag = random.random()<self.replace_rate
        new_item = item.copy()
        if replace_flag:
            for cnt in range(self.replace_cnt):
                for tried_cnt in range(10):
                    item_replace_idx = random.randint(0, len(new_item)-1)
                    random_keyidx = random.randint(0, len(self.seq_keys)-1)
                    replace_seq = self.seq_dict[self.seq_keys[random_keyidx]]
                    random_idx = random.randint(0, len(replace_seq)-1)
                    replace_it = replace_seq[random_idx]
                    # replace_it = {
                    #     "time":"0",
                    #     "content":"Reopen Block (BLK)",
                    #     "event_id":"0",
                    #     "template":"Reopen Block (BLK)",
                    #     "params":[],
                    #     "interval":100000,
                    #     'scaled_interval': 1,
                    #     'group_label': [1, 0],
                    # }
                    # break
                    if replace_it['template'] != new_item[item_replace_idx]['template']:
                        break
                if replace_it['template'] == new_item[item_replace_idx]['template']:
                    self.same_cnt += 1
                    self.same_cnt_all += 1                
                self.pass_replace_cnt += 1
                self.pass_replace_cnt_all += 1
                new_item[item_replace_idx] = replace_it
            self.random_replace_item[key]=(new_item, torch.FloatTensor([0, 1]))
            return self.random_replace_item[key]
        else:
            self.random_replace_item[key]=(new_item, torch.FloatTensor([1, 0]))
            return self.random_replace_item[key]


    def __len__(self):
        return len(self.seq_keys)

class BatchCollate():
    def __init__(self, pad_val=None) -> None:
        if pad_val:
            self.pad_val = pad_val
        else:
            self.pad_val = {
                "time":"0",
                "content":"log",
                "event_id":"0",
                "template":"log",
                "params":[],
                "interval":0,
                'scaled_interval': 0,
                'group_label': [1, 0],
            }
    
    def pad_tensor(self, it:list[dict], pad):
        pad_num = max(0, pad-len(it))
        it.extend([self.pad_val]* pad_num)
        return it
    
    def pad_replaceLabel(self, it, pad):
        pad_num = max(0, pad-len(it))
        it = torch.concat(
            [it, torch.FloatTensor([[1, 0]]).repeat(pad_num, 1)],
            dim=0
        )
        return it
    
    def __call__(self, batch_data) -> Any:
        batch_max_len = max([len(x) for (x, y, z, zz) in batch_data])
        batch_data = [(
                self.pad_tensor(x, pad=batch_max_len), 
                y,
                z, 
                zz, 
                # self.pad_replaceLabel(z, pad=batch_max_len), 
                ) for (x, y, z, zz) in batch_data]
        xs = [x[0] for x in batch_data]
        ys = torch.stack([x[1] for x in batch_data])
        zs = torch.stack([x[2] for x in batch_data])
        zzs = torch.stack([x[3] for x in batch_data])
        return xs, ys, zs, zzs


 