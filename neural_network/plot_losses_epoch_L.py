from utils import *

name=sys.argv[1]

with open(name, 'rb') as f:
    epoch_list, tr_loss_list, te_loss_list = pickle.load(f)

    plt.plot(epoch_list, tr_loss_list, label='training_loss')
    plt.plot(epoch_list, te_loss_list, label='testing_loss')
    plt.xlabel("epoch")
    plt.ylabel("losses")
    plt.legend()
    plt.savefig(name+'.png')
    plt.cla()
    plt.clf()

