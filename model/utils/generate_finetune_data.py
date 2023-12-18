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


generate_template()
