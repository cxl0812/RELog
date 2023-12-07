from transformers import BertTokenizer, BertConfig, AutoModel, BertModel, AutoModelForMaskedLM
import torch
from torch import nn
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogEncoder(nn.Module):
    def __init__(self, 
                 ):
        super().__init__()
        self.d_model = 768
        # this model is pretrained on all dataset, it is just in HDFS directory
        self.path = 'save_model/HDFS' 
        self.bert_tokenizer:BertTokenizer = BertTokenizer.from_pretrained(self.path, local_files_only=True)
        self.bert_config = BertConfig.from_pretrained(self.path, local_files_only=True)
        self.bert:AutoModel = AutoModel.from_pretrained(self.path, 
                                                        # config=self.bert_config, 
                                                        local_files_only=True,
                                                        # return_dict=True
                                                        ).to(device)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.template_embedding = {}


    def forward(self, batch_text_list):
        bert_outputs = []
        for text_list in batch_text_list:
            temp_bert_outputs_list = []
            for t in text_list:
                if t in self.template_embedding:
                    te = self.template_embedding[t].to(device)
                else:
                    # print("uncached: ", t)
                    inputs = self.bert_tokenizer(t, return_tensors="pt", padding=True)
                    inputs = inputs.to(device)
                    te = self.bert(**inputs)['last_hidden_state']# [:, 0, :]
                    te = torch.mean(te, dim=1)
                    self.template_embedding[t] = te.clone().detach().cpu()
                temp_bert_outputs_list.append(te)
            # print(temp_bert_outputs_list)
            temp_bert_outputs = torch.concat(temp_bert_outputs_list, dim=0)
            bert_outputs.append(temp_bert_outputs.unsqueeze(0))
        bert_outputs = torch.concat(bert_outputs, dim=0)

        return bert_outputs
        