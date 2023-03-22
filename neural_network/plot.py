from utils import *

results=[]
with open('study_L_20_90.pkl', 'rb') as f:
    # Load the contents of the pickle file into a Python object
    results.extend(pickle.load(f))

with open('study_L_100_170.pkl', 'rb') as f:
    # Load the contents of the pickle file into a Python object
    results.extend(pickle.load(f))

for i in range(len(results)):
    result=results[i]
    print(result)
    if (i==0):
        hide=''
    else:
        hide='_'
    plt.scatter(result[0], result[1], color='red' , label=hide+'training_loss')
    plt.scatter(result[0], result[2], color='blue', label=hide+'testing_loss')

plt.xlabel("lenght")
plt.ylabel("losses")
plt.legend()
plt.savefig('L_study.png')
plt.cla()
plt.clf()