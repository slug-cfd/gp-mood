from utils import *

with open('L_study.pkl', 'rb') as f:
    lenght_list, tr_loss_list, te_loss_list = pickle.load(f)

    plt.plot(lenght_list, tr_loss_list, label='training_loss')
    plt.plot(lenght_list, te_loss_list, label='testing_loss')
    plt.xlabel("lenght")
    plt.ylabel("losses")
    plt.legend()
    plt.savefig('L_study.png')
    plt.cla()
    plt.clf()

