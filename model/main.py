import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
import argparse
import numpy as np
from tqdm import tqdm
import torch
from LogEncoder import LogEncoder
from dataset import LogDataset, BatchCollate
from MainModel import BertTransformer
from torch.utils.data import  Dataset, DataLoader, random_split
from log_evolution import evolution_template, evolution_info
import pickle
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter  
import time 
from torch.optim.lr_scheduler import StepLR
import fasttext
import random
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--message', type=str, default="begin", help='message print to the beginning of log')
parser.add_argument('-l', '--load', type=str, default=None)
parser.add_argument('-s', '--save', type=str, default=None)
parser.add_argument('-ep', '--epoch', type=int, default=10)
parser.add_argument('-lr', '--learn_rate', type=float, default=1e-4)
parser.add_argument('-lrd', '--learn_rate_decay', type=float, default=0.9)
parser.add_argument('-rr', '--replace_rate', type=float, default=0)
parser.add_argument('-rn', '--replace_cnt', type=int, default=1)
parser.add_argument("-d", "--dataset", default="BGL", type=str)
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


args = parser.parse_args()
model_path = f'./save_model/{args.dataset}/mainModel/'
fasttext_model_path = f"save_model/{args.dataset}/param_fasttext_model.bin"
patch_path = f'../data/{args.dataset}/patch_'
epochs = args.epoch
val_epoch_interval = 1
batch_size = 100

# load different train dataset for Pre-Experiment
# when formal experiment, use the common '{args.dataset}_seq.pkl'
if args.mini:
    train_dataset_path = f'../data/{args.dataset}/mini_{args.dataset}_seq.pkl'
elif args.normal:
    train_dataset_path = f'../data/{args.dataset}/normal_{args.dataset}_seq.pkl'
elif args.abnormal:
    train_dataset_path = f'../data/{args.dataset}/abnormal_{args.dataset}_seq.pkl'
else:
    train_dataset_path = f'../data/{args.dataset}/{args.dataset}_seq.pkl'
print(f"train dataset path: {train_dataset_path}")
with open(train_dataset_path, 'rb') as f:
    seq_dict:dict = pickle.load(f)
    seq_key_list = list(seq_dict.keys())

# load label pickle
label_path = f'../data/{args.dataset}/label.pkl'
with open(label_path, 'rb') as f:
    label_dict = pickle.load(f)

# (*1)determine whether to add abnormal or evolution based on arguments,
# use the corresponding patch, update：seq_dict & label_dict, 
# the update of the normal/abnormal 'key list' will be carried out in subsequent code.
# 'key list' is the variables used to partition the train/val datasets
# (*1)here is the interval abnormal
interval_patch = {0:{'seq':{}, 'label':{}}}
if args.abnormal_interval_rate>0:
    with open(patch_path+'interval', 'rb') as f:
        interval_patch:dict = pickle.load(f)
        if args.abnormal_interval_rate not in interval_patch.keys():
            raise Exception("abnormal_interval_rate not in patch dict")
        seq_dict.update(interval_patch[args.abnormal_interval_rate]['seq'])
        label_dict.update(interval_patch[args.abnormal_interval_rate]['label'])
# (*1)here is the param abnormal
param_patch = {0:{'seq':{}, 'label':{}}}
if args.abnormal_param_rate>0:
    with open(patch_path+'param', 'rb') as f:
        param_patch:dict = pickle.load(f)
        if args.abnormal_param_rate not in param_patch.keys():
            raise Exception("abnormal_interval_rate not in patch dict")
        seq_dict.update(param_patch[args.abnormal_param_rate]['seq'])
        label_dict.update(param_patch[args.abnormal_param_rate]['label'])

# write the train and validate log
log_path = f'./log/{args.dataset}/MainModel/'+time.strftime("%Y-%m-%d-%H:%M:%S")+'/'
if not os.path.exists(log_path):
    os.makedirs(log_path)

generator = torch.Generator().manual_seed(42)
# if the arguments are specified, 
# use the random replacement strategy for pre training
if (args.replace_rate>0):
    train_rate = 0.9
    if(not args.not_random_pretrain_normal):
        # the other branch (the corresponding 'else') was only used in pre experiments 
        # and was not used in formal training.
        # that means, when random replace pretrain, 
        # use this branch, random split train and val dataset.
        train_idx, val_idx = random_split(
            seq_key_list, 
            [train_rate, 1-train_rate],
            generator=generator,
            )
        train_keys = [seq_key_list[i] for i in train_idx.indices]
        val_keys = [seq_key_list[i] for i in val_idx.indices]
    else:
        train_keys = [i for i in seq_key_list[:int(len(seq_key_list)*train_rate)]]
        val_keys = [i for i in seq_key_list[int(len(seq_key_list)*train_rate):]]
        random.shuffle(train_keys)
        random.shuffle(val_keys)
# do not use the random replacement strategy when finetune, 
# use the actual label insteadly
else:
    normal_seq_keys_list_file = f'../data/{args.dataset}/normal_{args.dataset}_seq_idx.pkl'
    abnormal_seq_keys_list_file = f'../data/{args.dataset}/abnormal_{args.dataset}_seq_idx.pkl'
    with open(normal_seq_keys_list_file, 'rb') as f:
        normal_keys_list:list = pickle.load(f)
    with open(abnormal_seq_keys_list_file, 'rb') as f:
        abnormal_keys_list:list = pickle.load(f)

    # (*1)update normal and abnormal key list based on patch
    # add new abnormal keys to abnormal_keys_list,
    # and remove new abnormal keys from the normal_keys_list
    abnormal_patch_keys = []
    abnormal_patch_keys.extend(list(interval_patch[args.abnormal_interval_rate]['seq'].keys()))
    abnormal_patch_keys.extend(list(param_patch[args.abnormal_param_rate]['seq'].keys()))
    abnormal_keys_list.extend(abnormal_patch_keys)
    abnormal_keys_list = list(set(abnormal_keys_list))
    normal_keys_list = [i for i in normal_keys_list if i not in abnormal_patch_keys]

    # take the top 90% of normal data and the random 50% of abnormal data as train dataset
    shuffled_abnormal_keys_list = abnormal_keys_list.copy()
    random.shuffle(shuffled_abnormal_keys_list)
    normal_train_rate = 0.9
    abn_train_rate = 0.5
    train_normal_keys_pool = [i for i in normal_keys_list[:int(len(normal_keys_list)*normal_train_rate)]]
    random.shuffle(train_normal_keys_pool)
    train_abnormal_keys_pool = [i for i in shuffled_abnormal_keys_list[:int(len(abnormal_keys_list)*abn_train_rate)]]
    abnormal_sample_cnt = int(len(abnormal_keys_list)*abn_train_rate)

    # adjust the rate of abnormal sample in the train dataset based on argument, 
    # 0 indicating no adjustment
    # calculate the normal_sample_cnt & abnormal_sample_cnt
    if args.abnormal_rate<=0:
        normal_sample_cnt = int(len(normal_keys_list)*normal_train_rate)
    else:
        normal_sample_cnt = int(abnormal_sample_cnt / args.abnormal_rate - abnormal_sample_cnt)
        if normal_sample_cnt > len(train_normal_keys_pool):
            normal_sample_cnt = len(train_normal_keys_pool)
            abnormal_sample_cnt = int(normal_sample_cnt / (1-args.abnormal_rate) - normal_sample_cnt)
    print(f"train-normal cnt: {normal_sample_cnt}, train-abnormal cnt: {abnormal_sample_cnt}")

    # determine the train_keys & val_keys based on normal_sample_cnt & abnormal_sample_cnt
    train_keys = train_normal_keys_pool[:normal_sample_cnt]
    # save train keys for analysis, not main process, not related to training 
    with open(f"../data/train_test_idx/{args.dataset}_train_normal_idx.pkl", 'wb') as f:
        pickle.dump(train_keys, f)
    with open(f"../data/train_test_idx/{args.dataset}_train_abnormal_idx.pkl", 'wb') as f:
        pickle.dump(train_abnormal_keys_pool[:abnormal_sample_cnt], f)
    train_keys.extend(
        train_abnormal_keys_pool[:abnormal_sample_cnt]
    )
    random.shuffle(train_keys)

    val_keys = [i for i in normal_keys_list[int(len(normal_keys_list)*normal_train_rate):]]
    # save val keys for analysis, not main process, not related to training 
    with open(f"../data/train_test_idx/{args.dataset}_test_normal_idx.pkl", 'wb') as f:
        pickle.dump(val_keys, f)
    with open(f"../data/train_test_idx/{args.dataset}_test_abnormal_idx.pkl", 'wb') as f:
        pickle.dump(shuffled_abnormal_keys_list[int(len(abnormal_keys_list)*abn_train_rate):], f)
    val_keys.extend(
        [i for i in shuffled_abnormal_keys_list[int(len(abnormal_keys_list)*abn_train_rate):]]
    )

    # (*1)here is the log evolution
    if args.template_evolution_rate>0:
        evolution_cnt = 0
        for k in val_keys:
            for it in seq_dict[k]:
                if random.random()<args.template_evolution_rate:
                    evolution_cnt += 1
                    it['template'] = evolution_template(it['template'])
        print(evolution_info)
        print(f"evolution seq cnt: {evolution_cnt}, val seq cnt: {len(val_keys)}")
    random.shuffle(val_keys)

# create train/val dataset/dataloader
train_dataset = LogDataset(
        seq_dict,
        label_dict,
        seq_keys=train_keys,
        replace_rate=args.replace_rate,
        replace_cnt=args.replace_cnt,
        max_len=200,
        )
train_dl = DataLoader(
    train_dataset, 
    collate_fn=BatchCollate(), 
    batch_size=batch_size,
    )
val_dataset = LogDataset(
        seq_dict,
        label_dict,
        seq_keys=val_keys,
        replace_rate=args.replace_rate,
        replace_cnt=args.replace_cnt,
        max_len=200
        )
val_dl = DataLoader(
    val_dataset, 
    collate_fn=BatchCollate(), 
    batch_size=batch_size
    )

# create or load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fastTextModel = fasttext.load_model(fasttext_model_path)
model = BertTransformer(
    LogEncoder(), # logEncoder use the pretrained bert model
    fastTextModel, # paramEncoder use the pretrained fastText model
    add_interval=not args.no_interval,
    add_param=not args.no_param,
    use_logEncoder=not args.no_logEncoder,
    freeze=not args.not_freeze,
    mask_rate=0.5,
).to(device)
if args.load:
    print(f"load model from '{args.load}'")
    model.load_state_dict(torch.load(model_path+args.load))

writer = SummaryWriter(log_path)
optimizer = torch.optim.Adam(model.parameters(),lr=args.learn_rate)
scheduler = StepLR(optimizer, step_size=int(5*len(train_dataset)//batch_size), gamma=args.learn_rate_decay)
loss_func = torch.nn.BCELoss()
train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []
log_interval = 1000//batch_size

with open(log_path+"log.txt", 'a') as f:
    f.write(args.message)
    f.write("\n")
    f.write(str(args._get_args()))
    f.write("\n")
    f.write(str(args._get_kwargs()))
    f.write("\n")

def cal_metric(label_class, pred_class):
    TN = sum((1-label_class)*(1-pred_class))
    TP = sum(label_class*pred_class)
    FN = sum(label_class*(1-pred_class))
    FP = sum((1-label_class)*pred_class)
    return {
        "f1": f1_score(label_class, pred_class),
        "p": precision_score(label_class, pred_class),
        "r": recall_score(label_class, pred_class),
        "acc": accuracy_score(label_class, pred_class),
        'spec': (TN)/(TN+FP),
        'TN': TN,
        'TP': TP,
        'FN': FN,
        'FP': FP, 
    }

# training and validating
for epoch in range(0, epochs+1):
    if(epoch!=0):
        print("lr: ", optimizer.param_groups[0]['lr'])
        model.train()
        train_epoch_loss = []
        loop_len = args.max_step_per_epoch if args.max_step_per_epoch>0 else len(train_dl)
        loop = tqdm(enumerate(train_dl), total=loop_len, mininterval=0.5)
        for idx,(data, label, replace_label, group_label) in loop:
            scheduler.step()
            label = label.to(device)
            replace_label = replace_label.to(device)
            outputs = model(data)
            optimizer.zero_grad()
            if args.replace_rate>0:
                loss = loss_func(outputs['replacement_prob'], replace_label)
            else:
                loss = loss_func(outputs['anomaly_prob'], label)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            loop.set_description(f'train Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item(), avg_loss=np.average(train_epoch_loss))
            if(idx%log_interval==0):
                writer.add_scalar("train_loss", 
                                loss.item(), 
                                global_step=(epoch*len(train_dl)+idx)*batch_size)
                writer.add_scalar("lr", 
                                optimizer.param_groups[0]['lr'], 
                                global_step=(epoch*len(train_dl)+idx)*batch_size)
            if(args.max_step_per_epoch>0 and idx>=args.max_step_per_epoch):
                break
        train_epochs_loss.append(np.average(train_epoch_loss))
        print("train loss: ", np.average(train_epoch_loss))
        train_dataset.same_cnt = 0
        train_dataset.pass_replace_cnt = 0
        
        if( epoch!=epochs-1 and epoch%val_epoch_interval!=0 ):
            # if val_epoch_interval!=1, then validate per val_epoch_interval epochs
            continue
    
    # validate
    model.eval()
    data_list = []
    valid_epoch_loss = []
    label_list = []
    replace_label_list = []
    group_label_list = []
    pred_list = []
    replace_pred_list = []
    loop = tqdm(enumerate(val_dl), total = len(val_dl), mininterval=0.5)
    for idx,(data, label, replace_label, group_label) in loop:
        label = label.to(device)
        replace_label = replace_label.to(device)
        data_list.extend(data)
        outputs = model(data)
        label_list.append(label.detach().cpu().numpy())
        replace_label_list.append(replace_label.detach().cpu().numpy())
        group_label_list.append(group_label.detach().cpu().numpy())
        pred_list.append(outputs['anomaly_prob'].detach().cpu().numpy())
        replace_pred_list.append(outputs['replacement_prob'].detach().cpu().numpy())
        if args.replace_rate>0:
            loss = loss_func(outputs['replacement_prob'], replace_label)
        else:
            loss = loss_func(outputs['anomaly_prob'], label)
        valid_epoch_loss.append(loss.item())
        if(idx%log_interval==0):
            writer.add_scalar("val_loss", 
                            loss.item(), 
                            global_step=(epoch*len(val_dl)+idx)*batch_size)
    valid_epochs_loss.append(np.average(valid_epoch_loss))
    print("val loss: ", np.average(valid_epoch_loss))

    # calculate and print the metric of random replace pretrain process and finetune process
    if args.replace_rate<=0:
        # this branch is not random replace pretrain process 
        # and is the finetune process
        label_list = np.concatenate(label_list, axis=0)
        group_label_list = np.concatenate(group_label_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        pred_class = np.argmax(pred_list, axis=1)
        label_class = np.argmax(label_list, axis=1)
        group_label_class = np.argmax(group_label_list, axis=1)
        print("sum pred , label : ", sum(pred_class), sum(label_class))
        metrics1 = cal_metric(label_class, pred_class)
        metrics1.update({
            "lr": optimizer.param_groups[0]['lr']
        })
        # only fast detection
        metrics2 = cal_metric(label_class, group_label_class)
        metrics2.update({
            "lr": optimizer.param_groups[0]['lr']
        })
        # use fast detection and model
        pred_class = np.max(np.stack([pred_class, group_label_class], axis=1), axis=1)
        metrics3 = cal_metric(label_class, pred_class)
        metrics3.update({
            "lr": optimizer.param_groups[0]['lr']
        })
        print("metric1: ", metrics1)
        print("metric2: ", metrics2)
        print("metric3: ", metrics3)
    else:
        # this branch is the random replace pretrain process 
        replace_label_list = np.concatenate(replace_label_list, axis=0)
        replace_pred_list = np.concatenate(replace_pred_list, axis=0)
        replace_pred_class = np.argmax(replace_pred_list, axis=1)
        replace_label_class = np.argmax(replace_label_list, axis=1)
        print("sum replace pred , label : ", sum(replace_pred_class), sum(replace_label_class))

        replace_metrics = {
            "f1": f1_score(replace_label_class, replace_pred_class),
            "p": precision_score(replace_label_class, replace_pred_class),
            "r": recall_score(replace_label_class, replace_pred_class),
            "acc": accuracy_score(replace_label_class, replace_pred_class),
            "lr": optimizer.param_groups[0]['lr'],
            "same_cnt": val_dataset.same_cnt,
            "replace_cnt": val_dataset.pass_replace_cnt,
            "same_cnt_all": val_dataset.same_cnt_all,
            "replace_cnt_all": val_dataset.pass_replace_cnt_all,
        }
        print("replace metric: ", replace_metrics)
        val_dataset.same_cnt = 0
        val_dataset.pass_replace_cnt = 0
    with open(log_path+"log.txt", 'a') as f:
        if args.replace_rate<=0:
            f.write(f"epoch: {epoch}, pure: ")
            f.write(str(metrics1))
            f.write("\n")
            f.write(f"epoch: {epoch}, pure group label: ")
            f.write(str(metrics2))
            f.write("\n")
            f.write(f"epoch: {epoch}, +group label: ")
            f.write(str(metrics3))
            f.write("\n")
            f.write("\n")
        else:
            f.write(f"epoch: {epoch}, replace metric ")
            f.write(str(replace_metrics))
            f.write("\n")
    if args.replace_rate<=0 and args.save and epoch>10 and (epoch+1)%1==0:
        print("save model.")
        torch.save(model.state_dict(), model_path+args.save+'-ep'+str(epoch))

# save model and data(for analysis)
if args.save:
    print("save model.")
    torch.save(model.state_dict(), model_path+args.save)
with open(log_path+"last_val_data.pkl", 'wb') as f:
    pickle.dump(data_list, f)
if args.replace_rate>0:
    with open(log_path+"last_replace_pred.pkl", 'wb') as f:
        pickle.dump(replace_pred_class, f)
    with open(log_path+"last_replace_label.pkl", 'wb') as f:
        pickle.dump(replace_label_class, f)

writer.close()