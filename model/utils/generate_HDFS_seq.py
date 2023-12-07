from random import sample
import random
import re
import pickle
import pandas as pd
import time
import numpy as np
import ast
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="HDFS", type=str)
parser.add_argument("-aia", "--add_interval_abnormal", default=1, type=float)
parser.add_argument("-apa", "--add_param_abnormal", default=1, type=float)

args = parser.parse_args()

hdfs_log_id_pattern = re.compile(r'blk_-?[0-9]+')

log_file = f'../data/{args.dataset}/{args.dataset}.log'
log_structed_file = f'../data/{args.dataset}/templates/{args.dataset}_ALL/{args.dataset}_ALL.txt_structured.csv'
log_seq_dict_file = f'../data/{args.dataset}/{args.dataset}_seq.pkl'
param_list_file = f'../data/{args.dataset}/param_list.txt'
mini_log_seq_dict_file = f'../data/{args.dataset}/mini_{args.dataset}_seq.pkl'
mini_normal_log_seq_dict_file = f'../data/{args.dataset}/mini_normal_{args.dataset}_seq.pkl'
template_seq_dict_file = f'../data/{args.dataset}/{args.dataset}_template_seq.log'
label_csv_path = f'../data/{args.dataset}/anomaly_label.csv'
label_path = f'../data/{args.dataset}/label.pkl'
normal_log_seq_dict_file = f'../data/{args.dataset}/normal_{args.dataset}_seq.pkl'
normal_seq_idx_list_file = f'../data/{args.dataset}/normal_{args.dataset}_seq_idx.pkl'
abnormal_seq_idx_dict_file = f'../data/{args.dataset}/abnormal_{args.dataset}_seq.pkl'
abnormal_seq_idx_list_file = f'../data/{args.dataset}/abnormal_{args.dataset}_seq_idx.pkl'
patch_path = f'../data/{args.dataset}/patch_'

NORMAL_LABEL = [1, 0]
ABNORMAL_LABEL = [0, 1]


def main():
    generate_log_seq()

    pass


HDFS_abnormal_group = [
    {"group_key1": "xxx", "group_key2":"xxx"},
    {"group_key1": "INFO", "group_key2":"dfs.DataNode$BlockReceiver"},
    {"group_key1": "WARN", "group_key2":"dfs.DataBlockScanner"},
    {"group_key1": "WARN", "group_key2":"dfs.DataNode$DataTransfer"},
    {"group_key1": "WARN", "group_key2":"dfs.PendingReplicationBlocks$PendingReplicationMonitor"},
]

def get_group_label(row, log_ids=None):
    for it in HDFS_abnormal_group:
        if it['group_key1']==row.Level and it['group_key2']==row.Component:
            if not log_ids is None:
                print("group_key==1, id: ", end='')
                print(log_ids)
            return ABNORMAL_LABEL
    return NORMAL_LABEL

def add_interval_abnormal_func(log_seq:list[dict]):
    new_seq = copy.deepcopy(log_seq)
    for i in range(4):
        idx = random.randint(0, len(new_seq)-1)
        new_seq[idx]['interval'] = random.randint(20, 2000)
    return new_seq

add_param_func_dict = {
    "EventId": lambda x: [x[0], x[1], random.randint(180, 30000)],
    "ba062512": lambda x : ["0.0.0.0", x[1], x[2]],
    "fe7f053c": lambda x : [random.randint(1000, 30000), x[1]],
    "c83fddbf": lambda x : [random.sample(['None', 'null', 'unknown', '/'], k=1)[0]],
}

def add_param_abnormal_func(log_seq:list[dict]):
    new_seq = copy.deepcopy(log_seq)
    flag = False
    for it in new_seq:
        if it['event_id'] in add_param_func_dict:
            it['params'] = [str(i) for i in add_param_func_dict[it['event_id']](it['params'])]
            flag = True
    if flag:
        return new_seq, flag
    else:
        return None, flag


camel_sub = re.compile(r"((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\\-\\_])[A-Za-z]|[\\-\\_])")
def camel_segment(s:str) -> str:
    if s.islower():
        return s
    if re.match(r"[A-Z]+(?:s|es)", s):
        return s.lower()
    return camel_sub.sub(r' \1', s).lower()

def cal_interval(time_a, time_b):
    time_a, time_b = '20'+time_a, '20'+time_b
    time_a, time_b = time.strptime(time_a, '%Y%m%d %H%M%S'), time.strptime(time_b, '%Y%m%d %H%M%S')
    time_a, time_b= int(time.mktime(time_a)), int(time.mktime(time_b))
    return (int(time_b)-int(time_a))

def generate_log_seq():
    mini_normal_size = 10000
    mini_abnormal_size = 10000
    label_df = pd.read_csv(label_csv_path)
    label_dict = {}
    normal_label_blkid_list = []
    abnormal_label_blkid_list = []
    for row in label_df.itertuples():
        blk_id = row.BlockId
        label = NORMAL_LABEL if (row.Label=='Normal') else ABNORMAL_LABEL
        label_dict[blk_id] = label

    log_seq_dict = {}
    log_seq_df = pd.read_csv(log_structed_file, 
                            #  nrows=80000,
                             dtype='str'
                             )
    cnt = 0
    for row in log_seq_df.itertuples():
        cnt+=1
        if cnt%100000==0:
            print(f"{cnt}/{len(log_seq_df)} done")
        log_id = getHDFSLogId(row.LogTokens)
        log_info = {
            "time":str(row.Date) + ' ' + str(row.Time),
            "content":camel_segment(row.Content),
            "event_id":row.EventId,
            "template":camel_segment(row.EventTemplate),
            "params":ast.literal_eval(row.ParameterList),
            "interval":0,
            "group_label": get_group_label(row, log_ids=None),
        }

        if not log_id in log_seq_dict:
            log_seq_dict[log_id] = [log_info]
        else:
            log_seq_dict[log_id].append(log_info)

        if (label_dict[log_id]==[1, 0]):
            normal_label_blkid_list.append(log_id)
        else:
            abnormal_label_blkid_list.append(log_id)
    normal_label_blkid_list = list(set(normal_label_blkid_list))
    abnormal_label_blkid_list = list(set(abnormal_label_blkid_list))
    print(f"abnormal/normal={len(abnormal_label_blkid_list)}/{len(normal_label_blkid_list)}")


    print("calculate interval")
    for k, v in log_seq_dict.items():
        v.sort(key=lambda v_item: v_item['time'])
        for idx, it in enumerate(v):
            if idx!=0:
                it['interval'] = cal_interval(
                    v[idx-1]["time"],
                    it["time"]
                )
    
    
    print("save params ")
    param_list = []
    for k, v in log_seq_dict.items():
        for it in v:
            param_list.extend(it['params'])
    with open(param_list_file, 'w') as f:
        for param in param_list:
            f.write(param)
            f.write('\n')

    print("add interval abnormal")
    interval_abnormal_patch = {}
    if args.add_interval_abnormal>0:
        for interval_abnormal_rate in [0.005, 0.01, 0.02, 0.03, 0.04]:
            patch_seq_dict = {}
            patch_label_dict = {}
            for k, v in log_seq_dict.items():
                if random.random()<interval_abnormal_rate:
                    patch_seq_dict[k] = add_interval_abnormal_func(log_seq_dict[k])
                    patch_label_dict[k] = ABNORMAL_LABEL
            interval_abnormal_patch[interval_abnormal_rate] = {}
            interval_abnormal_patch[interval_abnormal_rate]['seq'] = patch_seq_dict
            interval_abnormal_patch[interval_abnormal_rate]['label'] = patch_label_dict
    
    print("add param abnormal")
    param_abnormal_patch = {}
    if args.add_param_abnormal>0:
        for param_abnormal_rate in [0.005, 0.01, 0.02, 0.03, 0.04]:
            total_param_abnormal_cnt = param_abnormal_rate*len(log_seq_dict)
            cur_cnt = 0
            patch_seq_dict = {}
            patch_label_dict = {}
            for k, v in log_seq_dict.items():
                res, flag = add_param_abnormal_func(v)
                if(flag==True):
                    patch_seq_dict[k] = res
                    patch_label_dict[k] = ABNORMAL_LABEL
                    cur_cnt += 1
                    if cur_cnt>total_param_abnormal_cnt:
                        break
            param_abnormal_patch[param_abnormal_rate] = {}
            param_abnormal_patch[param_abnormal_rate]['seq'] = patch_seq_dict
            param_abnormal_patch[param_abnormal_rate]['label'] = patch_label_dict


    print("create mini dataset")
    mini_normal_blkid_list = sample(normal_label_blkid_list, mini_normal_size)
    mini_abnormal_blkid_list = sample(abnormal_label_blkid_list, mini_abnormal_size)
    mini_log_seq_dict = {k:log_seq_dict[k] for k in mini_normal_blkid_list}
    mini_log_seq_dict.update({k:log_seq_dict[k] for k in mini_abnormal_blkid_list})
    print(len(mini_normal_blkid_list), len(mini_normal_blkid_list), len(mini_log_seq_dict))
    print(len(set([*mini_abnormal_blkid_list, *mini_normal_blkid_list])))


    print("save datasets")
    with open(normal_log_seq_dict_file, 'wb') as f:
        pickle.dump({k:log_seq_dict[k] for k in normal_label_blkid_list}, f)
    with open(mini_log_seq_dict_file, 'wb') as f:
        pickle.dump(mini_log_seq_dict, f)
    with open(mini_normal_log_seq_dict_file, 'wb') as f:
        pickle.dump({k:log_seq_dict[k] for k in mini_normal_blkid_list}, f)

    with open(normal_seq_idx_list_file, 'wb') as f:
        pickle.dump(normal_label_blkid_list, f)
    with open(abnormal_seq_idx_dict_file, 'wb') as f:
        pickle.dump({k:log_seq_dict[k] for k in abnormal_label_blkid_list}, f)
    with open(abnormal_seq_idx_list_file, 'wb') as f:
        pickle.dump(abnormal_label_blkid_list, f)
    with open(log_seq_dict_file, 'wb') as f:
        pickle.dump(log_seq_dict, f)
    with open(label_path, 'wb') as f:
        pickle.dump(label_dict, f)
    with open(patch_path+'interval', 'wb') as f:
        pickle.dump(interval_abnormal_patch, f)
    with open(patch_path+'param', 'wb') as f:
        pickle.dump(param_abnormal_patch, f)
 

def generate_template_seq():
    template_seq_dict = {}
    with open(log_file, 'r') as lf:
        while(True):
            line = lf.readline()
            if not line:
                break
            log_id = getHDFSLogId(line)
            if not log_id in template_seq_dict:
                template_seq_dict[log_id] = [line]
            else:
                template_seq_dict[log_id].append(line)
    
    with open(template_seq_dict_file, 'wb') as f:
        pickle.dump(template_seq_dict, f)

def getHDFSLogId(log):
    find_res = hdfs_log_id_pattern.findall(log)
    if find_res:
        log_id = find_res[0]
    else:
        raise Exception("log line {} has no blk id.".format(log))
    return log_id
    

if __name__ == "__main__":
    main()