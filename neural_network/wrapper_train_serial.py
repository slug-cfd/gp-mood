from train_function import train
from utils import *

if (len(sys.argv)<5):
    print("ERROR, usage: python3.9 train_multiproc dataset model_name size gpu_id")
    sys.exit()

dataset_file=sys.argv[1]
model_name=sys.argv[2]
lenght=int(sys.argv[3])
gpu_id=int(sys.argv[4])

if __name__ == '__main__':

    train(lenght, dataset_file, model_name, softmax=True, gpu_id=gpu_id)
