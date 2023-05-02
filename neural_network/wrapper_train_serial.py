from train_function import train
from utils import *

if __name__ == '__main__':

    if (len(sys.argv)<6):
        print("ERROR, usage: python3.9 train_multiproc dataset model_name size loss_func gpu_id")
        sys.exit()

    dataset_file=sys.argv[1]
    model_name=sys.argv[2]
    lenght=int(sys.argv[3])
    loss_func=sys.argv[4]
    gpu_id=int(sys.argv[5])

    train(lenght, dataset_file, model_name, loss_func=loss_func, gpu_id=gpu_id)
