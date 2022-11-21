from torch import nn
import torch
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        #self.l1 = nn.Linear(2, 1024)
        # Output layer, 10 units - one for each digit
        #self.l2 = nn.Linear(00, 1024)
        self.l3=nn.Linear(1200,512)
        self.l4=nn.Linear(512,128)
        self.l5 = nn.Linear(128,32)
        self.l6 = nn.Linear(32,8)
        self.drop=nn.Dropout(0.1)
        self.output=nn.Linear(8,2)
        
        # Define sigmoid activation and softmax output 
        #self.sigmoid = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        #x=self.l1(x)
        #x=torch.relu(x)
        #x = self.l2(x)
        #x= self.d(x)
        #x = torch.relu(x)
       
        x = self.l3(x)
        #x= self.d(x)
        x = torch.relu(x)
        #x = torch.relu(x)
        #x= self.d(x)
        x = self.l4(x)
        x = torch.relu(x)
        #x= nn.Dropout(p=0.2)
        x = self.l5(x)
        x=torch.relu(x)
        #x = torch.relu(x)
        x=self.l6(x)
        x=torch.relu(x)
        
        x = self.output(x)
       
        return x
model = Network()

criterion=nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(),lr=0.0005)
optimizer2=torch.optim.Adam(model.parameters(),lr=0.001)

#total_steps=len(train_loader)

for epochs in range(100):
  train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=8,shuffle=True)
  for i,(samples,labels) in enumerate(train_loader):
    samples = samples.to(device)
    labels = labels.to(device)

    outputs=model(samples)
    loss=criterion(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if epochs%10==0:
    torch.save(model,'checkpoint'+str(epochs)+'.pth')
    print(loss,epochs)
    #print(test())

