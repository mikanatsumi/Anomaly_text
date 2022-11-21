import csv 
import numpy as np

items = []
labels=[]
l=0
temp=[]
mi=9999
ma=0
with open('/content/drive/MyDrive/ab.csv') as csvfile: 
    csvReader = csv.reader(csvfile)
    n=len(csvReader)
    for row in csvReader:
      #print(len(row))
      p=[int(float(v)) for v in row[1201:]]
      q=[int(float(v)) for v in row[1:1201]]
      #print(p)
      temp2=[]
      for j in range(1200):
        u=abs(p[j]-p[j+1200])
        temp2.append(u)
      mi = min(min(temp2),mi)
      ma = max(max(temp2),ma)
      #print(temp2)
      temp.append([v for v in temp2])
      labels.append(int(row[0]))
for j in range(n):
  items.append([(v-mi)/(ma-mi) for v in temp[j]])
labels=np.array(labels)
items=np.array(items)
#print(l)
labels = torch.LongTensor(labels)

items = torch.Tensor(items) # transform to torch tensor

print(items.shape)
print(labels.shape)
dataset=torch.utils.data.TensorDataset(items,labels)
train_set=torch.utils.data.TensorDataset(items,labels)
train_set, test_set = torch.utils.data.random_split(dataset, [(2*n)//3+n%3, n//3])
#for u,v in list(train_set):
  #print(v)
train_loader=torch.utils.data.DataLoader(dataset=train_set,batch_size=10,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_set,batch_size=1,shuffle=False)

#examples=iter(train_loader)
#samples,labels=examples.next()
#print(samples.shape,labels.shape)
