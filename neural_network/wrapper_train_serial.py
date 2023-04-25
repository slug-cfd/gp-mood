from train_function import train
from utils import *

if (len(sys.argv)<4):
    print("ERROR, usage: python3.9 train_multiproc dataset model_name size")
    sys.exit()

dataset_file=sys.argv[1]
model_name=sys.argv[2]
lenght=int(sys.argv[3])

if __name__ == '__main__':

    train(lenght, dataset_file, model_name, softmax=True)
