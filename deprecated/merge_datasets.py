from utils import *

#Get the file list and print it
print('')

file_list=sys.argv[1:-1]
filename=sys.argv[-1]+'.h5'

print(colors.yellow+"File list:"+colors.ENDC,file_list)
print('')
print('Saving in ',colors.yellow+filename+colors.ENDC)
print('')

#Amount of files
nbfiles=len(file_list)

#Initialize values for merged dataset size
NR0_merged=0
NR1_merged=0
size_merged=0

#list of sorted dataset for gathering
sorted_inputs=[]
sorted_labels=[]

#Statistics dictionnarues
problems={}
CFLs={}

for nfile,file in enumerate(file_list):

    print(colors.yellow+"/-- file #",nfile+1,"/",nbfiles,":", file, "--/"+colors.ENDC)

    #Get file and metadata
    f = h5py.File(file)
    problem=sum_char(f["problem"])
    print("problem", problem)
    method=sum_char(f["method"])
    print("method", method)
    CFL=sum_char(f["CFL"])
    print("CFL", CFL)
    print("lf", f["lf"][0])
    print("nf", f["nf"][0])

    #Get data
    inputs_numpy = np.array(f['inputs'])
    labels_numpy = np.array(f['labels'])

    #Sort data
    inputs_numpy_sorted, indices = np.unique(inputs_numpy, axis=0, return_index=True)
    labels_numpy_sorted= labels_numpy[indices]
    #Count R0 and R1 and size
    NR0=0
    NR1=0
    for item in labels_numpy_sorted:
        if (item[0]>item[1]):
            NR0+=1
        else:
            NR1+=1
    size=NR0+NR1 

    print("NR0 (sorted)", NR0)
    print("NR1 (sorted)", NR1)
    print("freq R0 (sorted)", format(NR0*100/size))
    print("size (sorted)",size)

    #Append sorted datasets to the lists
    sorted_inputs.append(inputs_numpy_sorted)
    sorted_labels.append(labels_numpy_sorted)

    #Add to the total sizes
    NR0_merged+=NR0
    NR1_merged+=NR1
    size_merged+=size

    #Gather statistics
    if problem in problems:
        problems[problem]+=size 
    else:
        problems[problem]=size
    
    if CFL in CFLs:
        CFLs[CFL]+=size 
    else:
        CFLs[CFL]=size
    
#Print info
print('')
print(colors.yellow+"Merged dataset info:"+colors.ENDC)
print("NR0 merged", NR0_merged)
print("NR1 merged", NR1_merged)
print("freq R0 merged", NR0_merged*100/size_merged)
print("size merged",size_merged)

print(colors.yellow+"Repartition per problem"+colors.ENDC)
for problem in problems.keys():
    print(problem, problems[problem], format(problems[problem]*100.0/size_merged))
print(colors.yellow+"Repartition per CFL"+colors.ENDC)
for CFL in CFLs.keys():
    print(CFL, CFLs[CFL], format(CFLs[CFL]*100.0/size_merged))

#Allocate merged dataset and store
merged_inputs=np.ones((size_merged,L), dtype=np.float32)*-666
merged_labels=np.ones((size_merged,2), dtype=np.float32)*-666

index=0
for nfile in range(0,nbfiles):
    size=np.shape(sorted_labels[nfile])[0]
    merged_inputs[index:index+size,:]=sorted_inputs[nfile][:,:]
    merged_labels[index:index+size,:]=sorted_labels[nfile][:,:]
    index=index+size

#Sanity checks
if (index-size_merged != 0):
    print("should be 0")
    sys.exit()
if (np.min(merged_inputs)<-665):
    print(np.min(merged_inputs))
    sys.exit()
if (np.min(merged_labels)<-665):
    print(np.min(merged_labels))
    sys.exit()

#Create a new HDF5 file
file = h5py.File(filename, "w")

#Write
file.create_dataset("inputs", data=merged_inputs)
file.create_dataset("labels", data=merged_labels)
file.create_dataset("size", data=size_merged)
file.create_dataset("NR0", data=NR0_merged)
file.create_dataset("NR1", data=NR1_merged)

# Close the file
file.close()

#Reduced dataset

#Create a new HDF5 file
file = h5py.File("reduced_"+filename, "w")
size_reduced=int(size_merged/2)
index=np.random.randint(0, size_merged, size_reduced)
#Write
file.create_dataset("inputs", data=merged_inputs[index])
file.create_dataset("labels", data=merged_labels[index])
file.create_dataset("size", data=size_reduced)
# Close the file
file.close()




