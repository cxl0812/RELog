from random import sample
import random
import numpy as np
import re
import pickle
import pandas as pd
import time
import ast
import copy
import argparse

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="BGL", type=str)
parser.add_argument("-aia", "--add_interval_abnormal", default=1, type=float)
parser.add_argument("-apa", "--add_param_abnormal", default=1, type=float)

args = parser.parse_args()

log_structed_file = f'../data/{args.dataset}/templates/{args.dataset}_ALL/{args.dataset}_ALL.txt_structured.csv'
log_seq_dict_file = f'../data/{args.dataset}/{args.dataset}_seq.pkl'
param_list_file = f'../data/{args.dataset}/param_list.txt'
normal_log_seq_dict_file = f'../data/{args.dataset}/normal_{args.dataset}_seq.pkl'
normal_seq_idx_list_file = f'../data/{args.dataset}/normal_{args.dataset}_seq_idx.pkl'
abnormal_seq_idx_dict_file = f'../data/{args.dataset}/abnormal_{args.dataset}_seq.pkl'
abnormal_seq_idx_list_file = f'../data/{args.dataset}/abnormal_{args.dataset}_seq_idx.pkl'
mini_log_seq_dict_file = f'../data/{args.dataset}/mini_{args.dataset}_seq.pkl'
mini_normal_log_seq_dict_file = f'../data/{args.dataset}/mini_normal_{args.dataset}_seq.pkl'
label_path = f'../data/{args.dataset}/label.pkl'
write_log_path = f'../data/{args.dataset}/write_log.txt'
patch_path = f'../data/{args.dataset}/patch_'
NORMAL_LABEL = [1, 0]
ABNORMAL_LABEL = [0, 1]


def main():
    generate_log_seq()

    pass

BGL_abnormal_group = [
    {"group_key1": "FATAL", "group_key2":"LINKCARD"},
    {"group_key1": "FATAL", "group_key2":"MMCS"},
    {"group_key1": "FAILURE", "group_key2":"BGLMASTER"}
]

def get_group_label(row, log_ids=None):
    for it in BGL_abnormal_group:
        if it['group_key1']==row.Level and it['group_key2']==row.Component:
            print("group_key==1, id: ", end='')
            print(log_ids)
            return ABNORMAL_LABEL
    return NORMAL_LABEL

def add_interval_abnormal_func(log_seq):
    new_seq = copy.deepcopy(log_seq)
    for i in range(40):
        idx = random.randint(1, len(new_seq)-1)
        new_seq[idx]['interval'] = random.randint(2000, 20000)
    return new_seq

add_param_func_dict = {
    "EventId": lambda x: [x[0], x[1], random.randint(180, 30000)],
    "d80bef44": lambda x: [random.randint(180, 30000)],
    "dc50ef19": lambda x: [random.randint(1000, 100000)*1000, random.randint(1000, 100000), x[2]],
    "5039b56c": lambda x: [random.randint(1000, 100000), random.randint(1000, 100000), x[2], x[3]],
}


def add_param_abnormal_func(log_seq):
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

def getBGLLogId(size, step):
    cnt = 0
    cur_seq_id_list = []
    next_seq_id = 1
    while(True):
        if(cnt%step==0):
            cur_seq_id_list.append(next_seq_id)
            next_seq_id += 1
        if(cnt>=size and (cnt-size)%step==0):
            cur_seq_id_list.pop(0)
        cnt += 1
        yield cur_seq_id_list

def generate_log_seq():
    try:
        normal_log_info = {}
        abn_log_info = {}

        mini_normal_size = 20000
        mini_abnormal_size = 20000
        write_log_interval = 1000
        write_log_list = []
        ids = getBGLLogId(100, 100)
        label_dict = {}
        normal_label_blkid_list = []
        abnormal_label_blkid_list = []
        label_dict = {}
        log_seq_dict = {}
        log_seq_df = pd.read_csv(log_structed_file, 
                                #  nrows=200000,
                                # skiprows=3500000,
                                dtype='str',
                                # names="LineId,Label,Timestamp,Date,Node,Time,NodeRepeat,Type,Component,Level,Content,EventId,EventTemplate,LogTokens,Params1,ParameterList".split(',')
                                )
        cnt = 0
        abn_cnt = 0
        for row in log_seq_df.itertuples():
            if pd.isna(row.Timestamp):
                continue
            cnt+=1
            if cnt%100000==0:
                print(f"{cnt}/{len(log_seq_df)} done")
            log_ids = next(ids)
            log_info = {
                "time": int(row.Timestamp),
                "content":camel_segment(row.Content),
                "event_id":row.EventId,
                "template":camel_segment(row.EventTemplate),
                "params":ast.literal_eval(row.ParameterList),
                "interval":0,
                "group_label": get_group_label(row, log_ids=log_ids),
            }
            log_label = NORMAL_LABEL if(row.Label=='-') else ABNORMAL_LABEL
            

            # Sequence template distribution analysis
            if log_label==NORMAL_LABEL:
                if log_info['template'] not in normal_log_info:
                    normal_log_info[log_info['template']] = {
                        'first_appear': cnt,
                        'count': 1,
                    }
                else:
                    normal_log_info[log_info['template']]['count'] += 1
            else:
                abn_cnt += 1
                if log_info['template'] not in abn_log_info:
                    abn_log_info[log_info['template']] = {
                        'first_appear': cnt,
                        'first_appert_in_abn': abn_cnt,
                        'count': 1,
                    }
                else:
                    abn_log_info[log_info['template']]['count'] += 1

            if cnt%write_log_interval==0:
                write_log_list.append(
                    [cnt/len(log_seq_df), cnt, abn_cnt, abn_cnt/cnt]
                )
            for log_id in log_ids:
                if not log_id in log_seq_dict:
                    log_seq_dict[log_id] = [log_info]
                else:
                    log_seq_dict[log_id].append(log_info)

                if not log_id in label_dict:
                    label_dict[log_id] = log_label
                elif(label_dict[log_id]==NORMAL_LABEL):
                    label_dict[log_id] = log_label

        
        # calculate interval and scaled_interval
        interval_list = []
        for k, v in log_seq_dict.items():
            v.sort(key=lambda v_item: v_item['time'])
            for idx, it in enumerate(v):
                if idx!=0:
                    it['interval'] = it["time"] - v[idx-1]["time"]
                    interval_list.append(it['interval'])
        
        mean_interval = np.mean(interval_list)
        sigma_interval = np.sqrt(np.var(interval_list))


        for k, v in log_seq_dict.items():
            v.sort(key=lambda v_item: v_item['time'])
            for idx, it in enumerate(v):
                if idx!=0:
                    it['scaled_interval'] = (it['interval'] - mean_interval) / sigma_interval
                else:
                    it['scaled_interval'] = 0
            
            assert log_seq_dict[k][0]['scaled_interval'] == 0
                    

        # add interval abnormal
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
        
        # add param abnormal
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
                if cur_cnt<total_param_abnormal_cnt:
                    raise Exception(f"Insufficient rate of injected parameter abnormal {param_abnormal_rate}, {cur_cnt}/{total_param_abnormal_cnt}")
                param_abnormal_patch[param_abnormal_rate] = {}
                param_abnormal_patch[param_abnormal_rate]['seq'] = patch_seq_dict
                param_abnormal_patch[param_abnormal_rate]['label'] = patch_label_dict
                  
                  
        for log_id, seq_label in label_dict.items():
            if(seq_label==NORMAL_LABEL):
                normal_label_blkid_list.append(log_id)
            elif(seq_label==ABNORMAL_LABEL):
                abnormal_label_blkid_list.append(log_id)
            else:
                raise Exception(f"Illegal label : '{str(seq_label)}'")
        print(f"abnormal/normal={len(abnormal_label_blkid_list)}/{len(normal_label_blkid_list)}")

        param_list = []
        for k, v in log_seq_dict.items():
            for it in v:
                param_list.extend(it['params'])
        with open(param_list_file, 'w') as f:
            f.write("cnt/len(log_seq_df), cnt, abn_cnt, abn_cnt/cnt \n")
            for param in param_list:
                f.write(param)
                f.write('\n')

        mini_abnormal_size = mini_normal_size = min([
            mini_abnormal_size, mini_normal_size,
            len(normal_label_blkid_list), len(abnormal_label_blkid_list)
        ])
        mini_normal_blkid_list = sample(normal_label_blkid_list, mini_normal_size)
        mini_abnormal_blkid_list = sample(abnormal_label_blkid_list, mini_abnormal_size)
        mini_log_seq_dict = {k:log_seq_dict[k] for k in mini_normal_blkid_list}
        mini_log_seq_dict.update({k:log_seq_dict[k] for k in mini_abnormal_blkid_list})
        print(len(mini_normal_blkid_list), len(mini_normal_blkid_list), len(mini_log_seq_dict))
        print(len(set([*mini_abnormal_blkid_list, *mini_normal_blkid_list])))

        with open(normal_log_seq_dict_file, 'wb') as f:
            pickle.dump({k:log_seq_dict[k] for k in normal_label_blkid_list}, f)
        with open(normal_seq_idx_list_file, 'wb') as f:
            pickle.dump(normal_label_blkid_list, f)
        with open(abnormal_seq_idx_dict_file, 'wb') as f:
            pickle.dump({k:log_seq_dict[k] for k in abnormal_label_blkid_list}, f)
        with open(abnormal_seq_idx_list_file, 'wb') as f:
            pickle.dump(abnormal_label_blkid_list, f)
        with open(mini_log_seq_dict_file, 'wb') as f:
            pickle.dump(mini_log_seq_dict, f)
        with open(mini_normal_log_seq_dict_file, 'wb') as f:
            pickle.dump({k:log_seq_dict[k] for k in mini_normal_blkid_list}, f)
        with open(log_seq_dict_file, 'wb') as f:
            pickle.dump(log_seq_dict, f)
        with open(label_path, 'wb') as f:
            pickle.dump(label_dict, f)
        with open(patch_path+'interval', 'wb') as f:
            pickle.dump(interval_abnormal_patch, f)
        with open(patch_path+'param', 'wb') as f:
            pickle.dump(param_abnormal_patch, f)
        with open(write_log_path, 'w') as f:
            for item in write_log_list:
                f.write(str(item))
                f.write('\n')
    except Exception as e:
        print(f'idx: {cnt}')
        raise e


if __name__=="__main__":
    main()