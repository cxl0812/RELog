# todo : 用全部数据集的模板来预训练

# export CUDA_VISIBLE_DEVICES=1
# export HF_DATASETS_OFFLINE=1
import json
from transformers import BertTokenizer, BertConfig, BertModel, AutoModelForMaskedLM, AdamW
from transformers import DataCollatorForLanguageModeling, AutoModelForNextSentencePrediction
from transformers import TrainingArguments, AutoModel
from datasets import load_dataset, load_metric
from transformers import Trainer, BertForPreTraining, TrainerCallback
from torch.utils.tensorboard import SummaryWriter
import torch
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pickle

UNCASED = '../bert-uncased/' 
bert_tokenizer:BertTokenizer = BertTokenizer.from_pretrained(UNCASED+'bert-base-uncased-vocab.txt', local_files_only=True)
bert_config = BertConfig.from_pretrained(UNCASED+'bert-base-uncased-config.json', local_files_only=True)
bert: BertModel = AutoModelForMaskedLM.from_pretrained(UNCASED+'bert-base-uncased-pytorch_model.bin', config=bert_config, local_files_only=True)


# path = 'save_model/test' 
# bert_tokenizer:BertTokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
# bert: BertModel = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True)
# bert_config = BertConfig.from_pretrained(path, local_files_only=True)

# # 导出templates -- utils/generate_finetune_data
# df = pd.read_csv('../data/templates/BGL.log_templates.csv')
# with open("../data/templates/BGL_finetune_input.txt", 'w') as f:
#     for i in df.itertuples():
#         f.write(i.EventTemplate + '\n')

def addToken():
    # 添加词库
    with open("../data/templates/BGL_finetune_input.txt") as f:
        templates_data = f.readlines()
    templates_vocab = [j for i in templates_data for j in i.split()]
    added_tk_num = bert_tokenizer.add_tokens(templates_vocab)
    print("added: ", added_tk_num)
    print("len of bert tokenizer: ", len(bert_tokenizer))
    bert.resize_token_embeddings(len(bert_tokenizer), pad_to_multiple_of=None)

class CustomCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None

    def on_init_end(self, args, state, control, model, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_train_end(self, args, state, control, model, tokenizer, **kwargs):
        if self.writer:
            self.writer.close()

    def on_step(self, args, state, control, model, inputs, outputs, optimizer, **kwargs):
        train_loss = outputs.loss
        global_step = state.global_step
        self.writer.add_scalar("train_loss", train_loss, global_step=global_step)



def MLMFinetune():
    # 构造MLM任务数据集
    def tokenize_function(examples):
        result = bert_tokenizer(examples["text"])
        if bert_tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    

    templates_dataset = load_dataset('text', 
                                     name='HDFS_finetune_input',
                                     data_files="../data/HDFS/templates/finetune_input.txt",
                                     )
    model_name = "HDFS"
    custom_callback = CustomCallback('log/HDFS')
    tokenized_datasets = templates_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=bert_tokenizer, mlm_probability=0.15)

    # # 查看mask效果
    # samples = [tokenized_datasets["train"][i] for i in range(2)]
    # for chunk in data_collator(samples)["input_ids"]:
    #     print(f"\n'>>> {bert_tokenizer.decode(chunk)}'")

    # 划分数据集
    train_size = 0.9
    test_size = 0.1

    downsampled_dataset = tokenized_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    print(downsampled_dataset)

    # 训练

    batch_size = 8
    logging_steps = len(downsampled_dataset["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        fp16=False,
        logging_steps=logging_steps,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
        tokenizer=bert_tokenizer,
        callbacks=[custom_callback],
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.train()
    trainer.save_model("save_model/"+model_name)
    bert_tokenizer.save_pretrained("save_model/"+model_name)
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


def generate_template_embedding():
    template_path = '../data/HDFS/templates/log_templates_new.csv'
    pickle_path = '../data/HDFS/templates/templates.embedding.pkl'

    path = 'save_model/HDFS' 
    bert_tokenizer:BertTokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
    bert: BertModel = AutoModel.from_pretrained(path, local_files_only=True)
    bert_config = BertConfig.from_pretrained(path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = bert.eval().to(device)
    template_df = pd.read_csv(template_path,sep=',', quotechar='"')
    embed_dict = {}
    cnt=0
    try:
        with torch.no_grad():
            for item in template_df.itertuples():
                cnt+=1
                template = item.NewEventTemplate
                inputs = bert_tokenizer(template, return_tensors="pt", padding=True)
                inputs = inputs.to(device)
                outputs = bert(**inputs)['last_hidden_state']
                embed_dict[template] = torch.mean(outputs, dim=1).cpu()
                torch.cuda.empty_cache()
    except Exception as e:
        print("cnt ", cnt)
        raise e
    with open(pickle_path, 'wb') as f:
        pickle.dump(embed_dict, f)
    

MLMFinetune()
generate_template_embedding()

