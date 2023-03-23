from utils import *

base_indexes=dict()

base_indexes[( 0 ,  0 )] = 0
base_indexes[(-1,   0 )] = 1
base_indexes[( 0 ,  1 )] = 2
base_indexes[( 1 ,  0 )] = 3
base_indexes[( 0,  -1 )] = 4
base_indexes[(-1,  -1 )] = 5
base_indexes[(-1 ,  1 )] = 6
base_indexes[( 1 ,  1 )] = 7
base_indexes[( 1 , -1 )] = 8
base_indexes[( 0,  -2 )] = 9
base_indexes[(-2 ,  0 )] = 10
base_indexes[( 0 ,  2 )] = 11
base_indexes[( 2 ,  0 )] = 12

rotations=[(1,1), (-1,1), (-1,-1), (1,-1)]
rotated_indexes=[[],[],[],[]]
nrot=0
for rotation in rotations:
    for key in base_indexes.keys():
        rotated_indexes[nrot].append(base_indexes[(key[0]*rotation[0], key[1]*rotation[1])])
    nrot=nrot+1

for rotated_index in rotated_indexes:
    print(rotated_index)

id=0
imx=1
imy=2
iE=3
nbvar=4

index_all_mx=range(imx, L-nbvar-1, nbvar)
index_all_my=range(imy, L-nbvar-1, nbvar)

#print( [i for i in index_all_mx] )
#print( [i for i in index_all_my] )

def rotate(data):
    rotated_data=[torch.ones(L)*-666,torch.ones(L)*-666,torch.ones(L)*-666,torch.ones(L)*-666]
    nrot=0
    for rotation in rotations:
        
        #Rotate the FV data
        for icell in range(13):
            #print(data.shape, rotated_indexes[nrot][icell]*nbvar,rotated_indexes[nrot][icell]*nbvar+nbvar-1)
            beg_rotated_data=icell*nbvar
            beg_data=rotated_indexes[nrot][icell]*nbvar
            lenght=nbvar
            rotated_data[nrot][beg_rotated_data:beg_rotated_data+lenght]=data[beg_data:beg_data+lenght]

        #Flip the x,y speed accordingly
        rotated_data[nrot][index_all_mx]*=rotation[0]
        rotated_data[nrot][index_all_my]*=rotation[1]
        #Get the CFL
        rotated_data[nrot][-1]=data[-1]
        #Get the normalisaiton factors (they dont flip, see test_norm.py)
        rotated_data[nrot][-nbvar-1:-1]=data[-nbvar-1:-1]
        #rotated_data[nrot]-=data
        nrot=nrot+1

    return rotated_data

def unit_resting_rotation(data0):

    data1=rotate(data0)     # 1: 11  -11 -1-1 1-1
    data2=rotate(data1[-1]) # 2: 1-1 -1-1 -11 11
    data3=rotate(data2[-1]) # 3: 11  -11 -1-1 1-1
    data4=rotate(data3[-1]) # 4: 1-1 -1-1 -11 11

    print("2 rotations should give 0 difference:",torch.max(data1[0]-data2[3]))
    print("2 rotations should give 0 difference:",torch.max(data1[0]-data3[0]))
    print("2 rotations should give 0 difference:",torch.max(data1[0]-data4[3]))

    print("2 rotations should give 0 difference:",torch.max(data1[1]-data2[2]))
    print("2 rotations should give 0 difference:",torch.max(data1[1]-data3[1]))
    print("2 rotations should give 0 difference:",torch.max(data1[1]-data4[2]))

    print("2 rotations should give 0 difference:",torch.max(data1[2]-data2[1]))
    print("2 rotations should give 0 difference:",torch.max(data1[2]-data3[2]))
    print("2 rotations should give 0 difference:",torch.max(data1[2]-data4[1]))

    print("2 rotations should give 0 difference:",torch.max(data1[3]-data2[0]))
    print("2 rotations should give 0 difference:",torch.max(data1[3]-data3[3]))
    print("2 rotations should give 0 difference:",torch.max(data1[3]-data4[0]))


def norm(x):
    xmin=torch.min(x)
    xmax=torch.max(x)

    xnorm=(x-0.5*(xmax+xmin))*(2./(xmax-xmin))
    F=torch.sign(xmin)*(xmax-xmin)

    return xnorm, F
    
def test_norm():

    data=torch.zeros((5))

    data[0]=1
    data[1]=-9
    data[2]=7
    data[3]=5
    data[4]=-10

    print(norm( data))
    print(norm(-data))