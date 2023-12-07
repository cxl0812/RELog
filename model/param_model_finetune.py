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

def test_model():
    model = fasttext.load_model(model_path)
    print(model[''])
    # for i in range(1000000):  # 2m1.519s
        # a = model['/user/root/rand/_logs/history/ip-10-250-19-102.ec2.internal_1226291400491_job_200811092030_0001_conf.xml.']
        # a = model['/user/root/rand/_logs/history/ip-10-250-19-102.ec2.internal_1226291400491_job_200811092030_0001_conf.xml.']


if __name__ == "__main__":
    get_model()
    # test_model()
    