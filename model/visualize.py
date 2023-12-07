import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
import argparse
import numpy as np
import torch
from LogEncoder import LogEncoder
from dataset import LogDataset, BatchCollate
from MainModel import BertTransformer
from torch.utils.data import  Dataset, DataLoader, random_split
from log_evolution import evolution_template, evolution_info
import pickle
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fasttext
from tqdm import tqdm
import random
random.seed(42)



parser = argparse.ArgumentParser()
parser.add_argument('-m', '--message', type=str, default="begin", help='message print to the beginning of log')
parser.add_argument('-l', '--load', type=str, default=f'./save_model/HDFS/mainModel/visual_-rr')
parser.add_argument('-s', '--save', type=str, default=None)
parser.add_argument('-ep', '--epoch', type=int, default=10)
parser.add_argument('-lr', '--learn_rate', type=float, default=1e-4)
parser.add_argument('-lrd', '--learn_rate_decay', type=float, default=0.9)
parser.add_argument('-rr', '--replace_rate', type=float, default=0)
parser.add_argument('-rn', '--replace_cnt', type=int, default=1)
parser.add_argument("-d", "--dataset", default="HDFS", type=str)
parser.add_argument('-ar', '--abnormal_rate', type=float, default=0)
parser.add_argument('-air', '--abnormal_interval_rate', type=float, default=0)
parser.add_argument('-apr', '--abnormal_param_rate', type=float, default=0)
parser.add_argument("-mini", "--mini", default=False, action='store_true')
parser.add_argument("-normal", "--normal", default=False, action='store_true')
parser.add_argument("-abnormal", "--abnormal", default=False, action='store_true')
parser.add_argument("-np", "--no_param", default=False, action='store_true')
parser.add_argument("-ni", "--no_interval", default=False, action='store_true')
parser.add_argument("-nle", "--no_logEncoder", default=False, action='store_true')
parser.add_argument("-nfz", "--not_freeze", default=False, action='store_true')
parser.add_argument("-not_random_pretrain_normal", "--not_random_pretrain_normal", default=False, action='store_true')
parser.add_argument('-ter', '--template_evolution_rate', type=float, default=0)
parser.add_argument('-spe', '--max_step_per_epoch', type=int, default=0)
parser.add_argument("-outfn", "--outfilename", default="test", type=str)



args = parser.parse_args()

# load model
model_path = args.load
fasttext_model_path = f"save_model/{args.dataset}/param_fasttext_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fastTextModel = fasttext.load_model(fasttext_model_path)
if str(args.load).find('-rr') != -1:
    args.no_logEncoder=True
if str(args.load).find('-i') != -1:
    args.no_interval=True
if str(args.load).find('-p') != -1:
    args.no_param=True  
if str(args.load).find('-ip') != -1:
    args.no_param=True
    args.no_interval=True
model = BertTransformer(
    LogEncoder(),
    fastTextModel,
    add_interval=not args.no_interval,
    add_param=not args.no_param,
    use_logEncoder=not args.no_logEncoder,
    freeze=not args.not_freeze,
    verbose=False
)

# if args.abnormal_param_rate>0:
#     print("load model apr")
#     model.load_state_dict(torch.load(f"save_model/{args.dataset}/mainModel/{args.dataset}_visual_ap"))
# elif args.abnormal_interval_rate>0:
#     print("load model air")
#     model.load_state_dict(torch.load(f"save_model/{args.dataset}/mainModel/{args.dataset}_visual_ai"))
# else:
print("load model")
model.load_state_dict(torch.load(args.load))

model = model.to(device)

# 读文件
train_dataset_path = f'../data/{args.dataset}/{args.dataset}_seq.pkl'
with open(train_dataset_path, 'rb') as f:
    seq_dict:dict = pickle.load(f)
    seq_key_list = list(seq_dict.keys())
label_path = f'../data/{args.dataset}/label.pkl'
with open(label_path, 'rb') as f:
    label_df = pickle.load(f)
normal_seq_keys_list_file = f'../data/{args.dataset}/normal_{args.dataset}_seq_idx.pkl'
abnormal_seq_keys_list_file = f'../data/{args.dataset}/abnormal_{args.dataset}_seq_idx.pkl'
with open(normal_seq_keys_list_file, 'rb') as f:
    normal_keys_list:list = pickle.load(f)
with open(abnormal_seq_keys_list_file, 'rb') as f:
    abnormal_keys_list:list = pickle.load(f)
patch_path = f'../data/{args.dataset}/patch_'
interval_patch = {0:{'seq':{}, 'label':{}}}
if args.abnormal_interval_rate>0:
    # 使用添加时间间隔异常的patch， update：seq_dict label_df
    with open(patch_path+'interval', 'rb') as f:
        interval_patch:dict = pickle.load(f)
        if args.abnormal_interval_rate not in interval_patch.keys():
            raise Exception("abnormal_interval_rate not in patch dict")
        seq_dict.update(interval_patch[args.abnormal_interval_rate]['seq'])
        label_df.update(interval_patch[args.abnormal_interval_rate]['label'])
param_patch = {0:{'seq':{}, 'label':{}}}
if args.abnormal_param_rate>0:
    # 添加参数异常的patch 同理
    with open(patch_path+'param', 'rb') as f:
        param_patch:dict = pickle.load(f)
        if args.abnormal_param_rate not in param_patch.keys():
            raise Exception("abnormal_interval_rate not in patch dict")
        seq_dict.update(param_patch[args.abnormal_param_rate]['seq'])
        label_df.update(param_patch[args.abnormal_param_rate]['label'])

normal_keys_list = [k for (k,v) in label_df.items() if v==[1, 0]]
abnormal_keys_list = [k for (k,v) in label_df.items() if v==[0, 1]]
print("len of normal/abnormal: ")
print(len(normal_keys_list))
print(len(abnormal_keys_list))

interval_patch_seq_key_list = interval_patch[args.abnormal_interval_rate]['seq'].keys()
param_patch_seq_key_list = param_patch[args.abnormal_param_rate]['seq'].keys()


alldata = {
    "param":{'data': []},
    "interval":{'data': []},
    "abnormal":{'data': []},
    "normal":{'data': []},
}

normal_cnt = 600
abn_cnt = 200
ai_cnt = min(len(interval_patch_seq_key_list), abn_cnt)
ap_cnt = min(len(param_patch_seq_key_list), abn_cnt)
for color, marker, name, small_key_list in (
    ('r', 'x', "param", random.sample(param_patch_seq_key_list, ap_cnt)),
    ('r', 'x', "interval", random.sample(interval_patch_seq_key_list, ai_cnt)),
    ('r', 'x', "abnormal", random.sample(abnormal_keys_list, abn_cnt)),
    ('g', '.', "normal", random.sample(normal_keys_list, normal_cnt)),
):
    alldata[name]['marker']=marker
    small_seq_dict = {k:seq_dict[k] for k in small_key_list}

    dataset = LogDataset(
            seq_dict,
            label_df,
            seq_keys=small_key_list,
            replace_rate=args.replace_rate,
            replace_cnt=args.replace_cnt,
            max_len=200,
            )
    dataloader = DataLoader(
        dataset, 
        collate_fn=BatchCollate(), 
        batch_size=2000,
        )
    model.eval()
    loop = tqdm(enumerate(dataloader), total = len(dataloader))
    for idx,(data, label, replace_label, group_label) in loop:
        outputs = model(data)
        # hiddens = torch.mean(outputs['outputs'], dim=-2).detach().cpu().numpy()
        pred = outputs['anomaly_prob'].detach().cpu().numpy()
        pred_class = np.argmax(pred, axis=1)
        print(label.shape)
        print(pred_class.shape)

        alldata[name]['data'] = [
            # torch.mean(outputs['LE_per_step'], dim=-2).detach().cpu().numpy(),
            # torch.mean(outputs['IE_per_step'], dim=-2).detach().cpu().numpy(),
            # torch.mean(outputs['PE_per_step'], dim=-2).detach().cpu().numpy(),
            # torch.mean(outputs['before_logEncoder'], dim=-2).detach().cpu().numpy(),
            # torch.mean(outputs['after_logEncoder'], dim=-2).detach().cpu().numpy(),
            torch.mean(outputs['outputs'], dim=-2).detach().cpu().numpy(),
        ]

plt.figure(dpi=200,figsize=(6,4))
# plt.tick_params(labelsize=8)
# plt.axis('off')
# for i, plt_name in enumerate(['template_embedding', 'interval_embedding', 'param_embedding', 'before_logEncoder', 'after_logEncoder', 'after_transforer2']):
for i, plt_name in enumerate(['hidden']):
    sample_data = np.concatenate(
        [alldata['param']['data'][i],
         alldata['interval']['data'][i],
         alldata['abnormal']['data'][i],
         alldata['normal']['data'][i]],
        axis=0
    )

    
    # print(hiddens.shape)
    mds = manifold.MDS(2, max_iter=300, n_init=4)
    vs = mds.fit_transform(sample_data)
    print(vs.shape)
    idx = 0
    alldata['param']['drawdata'] = vs[idx:idx+ap_cnt]; idx+=ap_cnt
    alldata['interval']['drawdata'] = vs[idx:idx+ai_cnt]; idx+=ai_cnt
    alldata['abnormal']['drawdata'] = vs[idx:idx+abn_cnt]; idx+=abn_cnt
    alldata['normal']['drawdata'] = vs[idx:idx+normal_cnt]; idx+=normal_cnt

    ax = plt.subplot(1, 1, i+1)
    # ax.set_title(str(plt_name))
    abn_is_labeled=False
    for k, v in alldata.items():
        if k=='normal':
            color = 'g'
            textlabel='normal'
        else:
            color = 'r'
            textlabel='abnormal'
            
        if k!='normal' and abn_is_labeled:
            ax.scatter(v['drawdata'][:, 0], v['drawdata'][:, 1], s=5, marker=v['marker'], color=color)
        else:
            abn_is_labeled=True
            ax.scatter(v['drawdata'][:, 0], v['drawdata'][:, 1], s=15, marker=v['marker'], color=color)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# plt.legend()
plt.savefig(f'fig/{args.outfilename}.png')


        # for subplt_idx, subplt_name, draw_data in (
        #     (1, 'IE_per_step', torch.mean(outputs['IE_per_step'], dim=-2).detach().cpu().numpy()),
        #     (2, 'PE_per_step', torch.mean(outputs['PE_per_step'], dim=-2).detach().cpu().numpy()),
        #     (3, 'before_logEncoder', torch.mean(outputs['before_logEncoder'], dim=-2).detach().cpu().numpy()),
        #     (4, 'after_logEncoder', torch.mean(outputs['after_logEncoder'], dim=-2).detach().cpu().numpy()),
        #     (5, 'hiddens', torch.mean(outputs['outputs'], dim=-2).detach().cpu().numpy()),
        # ):
        #     # 保存
        #     pass



