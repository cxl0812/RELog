import random
import json
import pandas as pd
import re
import os

template_file_dir = '../logparser-main/data/loghub_2k_corrected/'
# 生成 template_file_name 的 list
template_file_name_list = []
for root, dirs, files in os.walk(template_file_dir):
    for fn in files:
        if fn.endswith('templates_corrected.csv'):
            template_file_name_list.append(root+'/'+fn)


# template_file_name = '../data/HDFS/templates/HDFS_templates.csv'
output_csv_file_name = '../data/HDFS/templates/log_templates_new.csv'
output_file_name = '../data/HDFS/templates/finetune_input.txt'

# structed_file_name = '../data/HDFS/templates/HDFS.log_structured.csv'
# train_NSP_out_name = '../data/HDFS/templates/HDFS_train_BGL_finetune_input.txt'
# val_NSP_out_name = '../data/HDFS/templates/HDFS_val_BGL_finetune_input.txt'

camel_sub = re.compile(r"((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\\-\\_])[A-Za-z]|[\\-\\_])")
def camel_segment(s:str) -> str:
    if s.islower():
        return s
    if re.match(r"[A-Z]+(?:s|es)", s):
        return s.lower()
    return camel_sub.sub(r' \1', s).lower()


def generate_template():
    all_df = pd.DataFrame()
    for template_file_name in template_file_name_list:
        df = pd.read_csv(template_file_name, on_bad_lines='warn')
        df['NewEventTemplate'] = df.apply(lambda row:camel_segment(row.EventTemplate), axis=1)
        all_df = pd.concat([all_df, df])
    with open(output_file_name, 'w') as f:
        for i in all_df.itertuples():
            s = i.NewEventTemplate
            f.write(s + '\n')
    all_df.to_csv(output_csv_file_name)

def generate_pair():
    # 生成序列对 -> NSP
    df = pd.read_csv(structed_file_name)
    train_size=100000
    val_size = 10000
    fake_rate = 0.5
    train_list = get_NSP_item(df, train_size, fake_rate)
    # val_list = get_NSP_item(df, val_size, fake_rate)

    with open(train_NSP_out_name, 'w') as f:
        for i in train_list:
            f.write(json.dumps(i))
            f.write('\n')
    
    # with open(val_NSP_out_name, 'w') as f:
    #     for i in val_list:
    #         f.write(json.dumps(i))
    #         f.write('\n')
        

def get_NSP_item(df, size, fake_rate):
    data_list = []
    for i in range(size):
        idx = random.randint(0, df.shape[0]-1)
        # text_a 确保是一条正常的日志
        while(df.iloc[idx].Label!="-"):
            idx = random.randint(0, df.shape[0])
        text_a = camel_segment(df.iloc[idx].EventTemplate)

        if random.random()>fake_rate:
            # 直接取text_a的下一条日志，label为1（除非下一条日志是异常日志）
            text_b = camel_segment(df.iloc[idx+1].EventTemplate)
            label = 1 if df.iloc[idx+1].Label=="-" else 0
        else:
            # 由于数量足够大，基本就算不是相邻的了
            idx_b = random.randint(0, df.shape[0])
            text_b = camel_segment(df.iloc[idx_b].EventTemplate)
            label = 0
        data_list.append({
            "text_a": text_a,
            "text_b": text_b,
            "label": label,
        })
    return data_list

generate_template()
# generate_pair()