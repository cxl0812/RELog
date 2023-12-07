from transformers import BertTokenizer, BertConfig, BertModel, AutoModelForMaskedLM, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import Trainer
import torch
import math
import numpy as np 
from torch import nn
import pickle
from LogEncoder import LogEncoder
import fasttext
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.template_embed_matrix = nn.Parameter(torch.rand(self.in_dim, self.out_dim))

    def forward(self, x):
        return torch.matmul(x, self.template_embed_matrix)


class BertTransformer(nn.Module):
    def __init__(self, 
                 logEncoder,
                 paramEncoder,
                 max_len=512,
                 mask_rate=0.5,
                 add_param=True,
                 add_interval=True,
                 use_logEncoder=True,
                 freeze=True,
                 verbose=False,
                 ):
        super().__init__()
        self.add_param = add_param
        self.add_interval = add_interval
        self.use_logEncoder = use_logEncoder
        self.freeze = freeze
        self.verbose = verbose
        print(f"param: {self.add_param}, interval: {self.add_interval}, logEncoder: {self.use_logEncoder}")
        self.d_model = 768  
        self.d_model_each = 256
        self.d_model_all = self.d_model_each*(1+self.add_param+self.add_interval)
        self.position_encoding_tensor = self.positional_encoding(max_len, self.d_model_all).to(device)
        self.mask_rate = mask_rate
        self.log_encoder = logEncoder
        self.param_encoder = paramEncoder
        self.template_adaptor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model_each),
        )
        self.param_adaptor = nn.Sequential(
            nn.Linear(self.param_encoder.get_dimension(), self.d_model_each),
        )
        self.interval_adaptor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model_each),
        )
        
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.d_model_all,
                                        nhead=4,
                                        batch_first=True)
        self.transformer_encoder_2 = nn.TransformerEncoderLayer(d_model=self.d_model_all,
                                        nhead=4,
                                        batch_first=True)

        self.bce_loss = nn.BCELoss()
        self.hidden2replacementProb = nn.Sequential(
            nn.Linear(self.d_model_all, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        self.hidden2prob = nn.Sequential(
            nn.Linear(self.d_model_all, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, d):
        # basic embedding : log (template) embedding
        LE_per_step = self.log_encoder([[iit['template'] for iit in it] for it in d])
        LE_per_step = self.template_adaptor(LE_per_step)

        # calculate and concat param embedding and interval embedding if the arguments specifies
        if(self.add_param):
            PE_pre_step = self.get_param_embedding([[iit['params'] for iit in it] for it in d])
            PE_pre_step = self.param_adaptor(PE_pre_step)
        if(self.add_interval):
            interval = torch.Tensor([[1/(iit['interval']+1) for iit in it] for it in d]) \
                            .unsqueeze(-1)\
                            .to(device)
            IE_per_step = self.interval_adaptor(interval)
        embedding = LE_per_step
        if self.add_param:
            embedding = torch.concat([embedding, PE_pre_step], dim=-1)
        if self.add_interval:
            embedding = torch.concat([embedding, IE_per_step], dim=-1)

        # Save intermediate variables for verbose output 
        before_logEncoder = embedding.clone()

        # Determine whether to use log_encoder(self.transformer_encoder) based on arguments
        if(self.use_logEncoder):
            embedding = embedding + self.position_encoding_tensor[:, :embedding.size(1), :]
            hidden = self.transformer_encoder(embedding)
            if self.freeze:
                hidden = hidden.detach()
        else:
            hidden = embedding

        # Save intermediate variables for verbose output 
        after_logEncoder = hidden.clone()

        # calculate replacement_prob(for pretrain) and anomaly_prob(for finetune) respectively
        replacement_prob = torch.mean(hidden, dim=-2)
        replacement_prob = self.hidden2replacementProb(replacement_prob)

        hidden = hidden + self.position_encoding_tensor[:, :hidden.size(1), :]
        hidden = self.transformer_encoder_2(hidden)
        anomaly_prob = torch.mean(hidden, dim=-2)
        anomaly_prob = self.hidden2prob(anomaly_prob) 

        if not self.verbose:
            return {
                "outputs": hidden,
                "anomaly_prob":anomaly_prob,
                "replacement_prob": replacement_prob,
            }
        else:
            return {
                "outputs": hidden,
                "anomaly_prob":anomaly_prob,
                "replacement_prob": replacement_prob,
                "LE_per_step": LE_per_step, 
                "IE_per_step": IE_per_step,
                "PE_per_step": PE_pre_step,
                "before_logEncoder": before_logEncoder,
                "after_logEncoder": after_logEncoder,
            }

    def get_param_embedding(self, batch_multiitems_param_list:list):
        """get the param embedding of a batch of param list

        Args:
            batch_multiitems_param_list (list): a batch of param list, each param list is 
                a list which contains params of a log line.

        Returns:
            Tensor: a batch of embedding, each embedding corresponds to a log line
        """
        batch_embedding = []
        for multiitems_param_list in batch_multiitems_param_list:
            item_param_embedding = []
            for item_param_list in multiitems_param_list:
                if len(item_param_list)==0:
                    param_embedding_list = torch.Tensor(np.array(
                        [self.param_encoder[i] for i in [""]]))
                else:
                    param_embedding_list = torch.Tensor(np.array(
                        [self.param_encoder[i] for i in item_param_list]))
                item_param_embedding.append(param_embedding_list.mean(dim=0))
            item_param_embedding = torch.stack(item_param_embedding, dim=0) 
            batch_embedding.append(item_param_embedding)
        batch_embedding = torch.stack(batch_embedding)
        return batch_embedding.to(device)

    def positional_encoding(self, max_seq_len, d_model):
        P = torch.zeros((1, max_seq_len, d_model))
        X = torch.arange(max_seq_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        P[:, :, 0::2] = torch.sin(X)   
        P[:, :, 1::2] = torch.cos(X)
        return P

    def cal_loss(self, outputs, replace_loc):
        return self.bce_loss(outputs, replace_loc)

