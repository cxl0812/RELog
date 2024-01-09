# R2log
R2Log: Robust and Real-time Anomaly Detection Based on Complete Logs for Large-scale Systems

---
## Project Structure
The file structure is as belows:

```
.
├── bert-uncased
│   └── bert model files
├── data
│   ├── BGL
│   │   ├── templates
│   │   └── other files generate by ```generate_BGL_seq.py```
│   ├── HDFS
│   │   ├── templates
│   │   └── other files generate by ```generate_HDFS_seq.py```
├── model
│   ├── save_model
│   │   ├── BGL
│   │   │   ├── mainModel
│   │   │   └── other files
│   │   ├── HDFS
│   │   │   ├── mainModel
│   │   │   └── other files
│   └── model python code
└── README.md
```
We have uploaded the saved model and origin dataset to the cloud storage. Please download and decompress through this link if needed:
https://pan.baidu.com/s/1kwEQz1rKUYaOKMIV8mu1ZA?pwd=auhz 

---
## Requirements
This project is base on Python 3.10.12. Please run the command to install other key packages.
```
pip install -r requirements.txt
```

---
## Run the project
This project contains 3 main steps: (1) generate the datasets, (2) pretrain the templateEncoder and paramEncoder, (3) run the train&test process.


### Step1: generate the datasets
After the origin dataset is ready, please run the following script to complete the generation and partitioning of the dataset.
```
cd model
python utils/generate_BGL_seq.py
python utils/generate_HDFS_seq.py
# python utils/generate_finetune_data.py   # this script is used to generate `log_templates_new.csv` and `finetune_input.txt`, which are concatenated from the templates on  `loghub`. They have been generated and included in the data directory. There is no need to run it again.
```
After these script, the datasets used in different experiments will be generated in directory data/BGL and data/HDFS.

### Step2: pretrain the templateEncoder and paramEncoder
To pretrain the bert model as templateEncoder, and Fasttest model as param Encoder, please run the following script.
```
cd model  # if not in model dir
python finetune.py
python param_model_finetune.py
```

### Step3: run the train&test process
To carry out training and testing work, please run the following scripts. For information of arguments, please view the information through ```python main.py -h``` .
```
# pretrain the logEncoder use random replace
python main.py -m "session information write to log, optional" -d BGL -s "BGL-test-8-1" -ep=20 -abnormal -rr=0.2 -rn=4 -nfz

# finetune on actual label
python main.py -d BGL -l "BGL-test-8-1" -s "BGL-test-8-2" -ep=20 

```


