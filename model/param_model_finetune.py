import fasttext
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="HDFS", type=str)

args = parser.parse_args()


param_list_file = f'../data/{args.dataset}/param_list.txt'
model_path = f"save_model/{args.dataset}/param_fasttext_model.bin"
print(param_list_file)

def get_model():
    model = fasttext.train_unsupervised(
        param_list_file,
        dim=100,
        epoch=30,
        minn=1,
        maxn=6,   
    )

    model.save_model(model_path)

    print(len(model.words))


if __name__ == "__main__":
    get_model()
    
