# from transformers import BertTokenizer, BertConfig, AutoModel, BertModel, AlbertTokenizer, AlbertConfig, AlbertModel, AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
from torch import nn
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogEncoder(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.d_model = 768
        self.max_len = 512
        # this model is pretrained on all dataset, it is just in HDFS directory
        self.model_path = 'save_model/HDFS' 
        # self.model_path: str = '../albert-base-v2'
        # self.model_path: str = '../bert-uncased'
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        self.model:AutoModel = AutoModel.from_pretrained(self.model_path, 
                                                        config=self.config, 
                                                        local_files_only=True,
                                                        # return_dict=True
                                                        ).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.template_embedding = {}


    def forward(self, batch_text_list: list) -> torch.Tensor:
        all_outputs = []
        for text_list in batch_text_list:
            seq_outputs = []
            for t in text_list:
                if t in self.template_embedding:
                    te = self.template_embedding[t].to(device)
                else:
                    # print("uncached: ", t)
                    inputs = self.tokenizer(t, return_tensors="pt", padding=True, max_length=self.max_len, truncation=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    te = self.model(**inputs)['last_hidden_state']# [:, 0, :]
                    te = torch.mean(te, dim=1)
                    self.template_embedding[t] = te.clone().detach().cpu()
                seq_outputs.append(te)
            # print(temp_bert_outputs_list)
            seq_tensor = torch.concat(seq_outputs, dim=0).unsqueeze(0)
            all_outputs.append(seq_tensor)

        return torch.concat(all_outputs, dim=0)
        
