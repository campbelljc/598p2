import pickle
import numpy as np

def p_load(file):
    with open("processed_data/" + file, 'rb') as infile:
        return pickle.load(infile)
        

# single row of word names corresponding to below word columns

word_count_matrix = p_load('__.dat')  # Put in the file name here .dat

# each row is an interview excerpt
# each column is a word
# each cell represents the number of times a word occurs in an interview

#print (word_count_matrix.shape)

data = word_count_matrix[:100,0:word_count_matrix.shape[1]-1]
target = word_count_matrix[:100,-1]

# Checking for particular class in target   

class0=target==0
class1=target==1
class2=target==2
class3=target==3

# Grouping targets except for particular class

class0i=target!=0
class1i=target!=1
class2i=target!=2
class3i=target!=3

# Grouping data with particular class

class0_data=data[class0]
class1_data=data[class1]
class2_data=data[class2]
class3_data=data[class3]

# Grouping data with particular classes

class0_datai=data[class0i]
class1_datai=data[class1i]
class2_datai=data[class2i]
class3_datai=data[class3i]

# Forming a list 

class_data=[class0_data,class1_data,class2_data,class3_data]
class_datai=[class0_datai,class1_datai,class2_datai,class3_datai]
class_data=np.array(class_data)
class_datai=np.array(class_datai)


# calculate the probability for each each Class

py0=len(target[class0])*1.00/len(target)
py1=len(target[class1])*1.00/len(target)
py2=len(target[class2])*1.00/len(target)
py3=len(target[class3])*1.00/len(target)

#print "Total probability check:  ",py0+py1+py2+py3

#for each class
py=[py0,py1,py2,py3] #List of class prob

#calculating likelyhood

p= [ [ 0 for i in range(len(py)) ] for j in range(len(data[1][:])) ]
pi=[ [ 0 for i in range(len(py)) ] for j in range(len(data[1][:])) ]
print np.array(p).shape
for feat in range(len(data[1][:])):
    for clas in range(len(py)):
        temp=class_data[clas]
        tempi=class_datai[clas]
        p[feat][clas]=(sum(temp[:,feat])+1.00)/(len(temp[:,feat])+2)   # Laplacian smoothing P(xj = i|c = k)
        pi[feat][clas]=(sum(tempi[:,feat])+1.00)/(len(tempi[:,feat])+2) # Laplacian smoothing P(xj = i|c = k)

p=np.array(p)
pi=np.array(pi)

# validation
def getprobval(f,c,k,o): # returns the probability value based on the args
    if o==1:
        if k==0 or p[f,c]==0:
            return (1.00)  
        else:
            return p[f,c]
    if o==2:
        if k==0 or pi[f,c]==0:
            return (1.00)  
        else:
            return pi[f,c]

print "Staring Validation"

val_data=word_count_matrix[0:10000,0:word_count_matrix.shape[1]-1] # Selecting validation set 
test_target=word_count_matrix[0:10000,-1] 
est_target=[]  #Estimated target values will be appended here 
for vect in val_data:
    tp=np.array([py0,py1,py2,py3])
    tpi=np.array([(1-py0),(1-py1),(1-py2),(1-py3)])
    for c in range(len(py)):
        for feat in range(len(vect)):
            
            tp[c]=tp[c]*getprobval(feat,c,vect[feat],1)
            
            tpi[c]=tpi[c]*getprobval(feat,c,vect[feat],2)
            
        tp[c]=tp[c]*1.00/(tp[c]+tpi[c])          
    
    # Estimating argmax of probs
    est_target.append(np.argmax(tp))
    
# Accuracy

Acc=0
for it in range(len(test_target)):
    if est_target[it]==test_target[it]:
        Acc=Acc+1
Acc=Acc*1.00/len(test_target)
print Acc
