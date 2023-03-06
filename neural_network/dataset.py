import os

class data_point():

    def __init__(self, line):

        columns = line.split()
        line = [float(column) for column in columns]
        L=len(line)
        self.FV_data=line[0:L-3]
        self.R=line[-1]
        self.CFL=line[-2]


# Define the path to the text file

path = '../gp-mood-code/'
paths = []

for file_name in os.listdir(path):
    if "trimmed" in file_name:
        paths.append(path+file_name)


# Initialize an empty list to store the data
data0 = []
data1 = []

N0=0
N1=0

for path in paths:
    with open(path, 'r') as file:

        # Loop through each line of the file
        for line in file:
            data=data_point(line)

            if (data.R==0):
                N0+=1
                data0.append(data)
            else:
                N1+=1
                data1.append(data)
        print(path)
        print((100*N0)/(N0+N1), (100*N1)/(N0+N1))

print(len(data0), len(data1))
    



