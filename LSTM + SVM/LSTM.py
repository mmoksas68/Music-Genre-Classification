import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from performance_metrics import report_performance
from torch.autograd import Variable
import matplotlib.pyplot as plt

# %%  EXTRACT SOUND WAVES FROM SONGS
def extract_1d_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    data = []
    for g in genres:     
        for filename in os.listdir(f'Preprocess/rescaled400_daha_net/{g}'):
            songname = f'Preprocess/rescaled400_daha_net/{g}/{filename}'
            temp = plt.imread(songname)[:,:,:3]
            temp.transpose(1,0,2)
            data.append(np.reshape(temp,(400*400*3)))
    return data

def shuffle(data, classes):
    dataset = np.r_['1', data, classes]
    np.random.shuffle(dataset)
    n, m = np.shape(dataset)
    train_data = dataset[:700].astype(dtype='float32')
    validation_data = dataset[700:800].astype(dtype='float32')
    test_data = dataset[800:].astype(dtype='float32')
    return train_data, validation_data, test_data

def construct_labels():
    labels = []
    numeration = [0,1,2,3,4,5,6,7,8,9]
    for g in numeration:
        for i in range(100):
            labels.append(g)
    return np.reshape(labels,(1000,1))



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 

        return out

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

    
# %% set train, validation, and test data

data = extract_1d_data()
labels = construct_labels()
train_data, validation_data, test_data = shuffle(data, labels)

# %% prepare dataset for model
train_data = np.r_['0', train_data, validation_data]

# parameters 
batch_size = 64
num_epochs = 120


train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size = batch_size,
                                           shuffle=True, num_workers=2)

validation_loader = torch.utils.data.DataLoader(dataset=validation_data, 
                                               batch_size = 100,
                                               shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size = 200,
                                          shuffle=False, num_workers=2)


# %% initializing parameters and the model
seq_dim = 400
input_dim = 400*3
hidden_dim = 400
layer_dim = 2
output_dim = 10

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# %% training the model

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


train_accs = []
train_losses = []
validation_accs = []

best_outputs = []
max_test_acc = 0

iter = 0
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    model.train() 
    iter += 1

    c = 0
    for i, data in enumerate(train_loader):
        c += 1
        m = np.shape(data)[1]
        label = data[:,m-1]
        label = label.to(dtype=torch.int64)
        
        data = data[:,:m-1]
        data = np.reshape(data,(batch_size, 1, -1))
        images, labels = Variable(data), Variable(label)
        
        # Load images as a torch tensor with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images.to(device)).to(device)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs.to('cpu'), labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs.to('cpu'), labels, batch_size)
  
    
    # Calculate Validation Accuracy         
    correct = 0
    total = 0
    # Iterate through validation dataset
    for i, data in enumerate(validation_loader):
        m = np.shape(data)[1]
        label = data[:,m-1]
        label = label.to(dtype=torch.int64)
        
        data = data[:,:m-1]
        data = np.reshape(data,(batch_size, 1, -1))
        images, labels = Variable(data), Variable(label)
        # Resize images
        images = images.view(-1, seq_dim, input_dim).requires_grad_()

        # Forward pass only to get logits/output
        outputs = model(images.to(device)).to(device)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted.to('cpu') == labels).sum()

    accuracy = 100 * correct / total
            
    model.eval()
    train_losses.append(train_running_loss/c)
    train_accs.append(train_acc/c)
    validation_accs.append(accuracy)
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' 
      %(epoch, train_running_loss / c, train_acc/c))
    
    print('Epoch:  %d Validation Accuracy: %.2f' 
      %(epoch, accuracy))
    
    print('-------------------------------------')
    if(train_acc/c > 87 or accuracy > 54):
        break

# %% Plotting Training Accuracy and Validation Accuracy vs Epoch
#    Plotting Training Loss vs Epoch
epochs = range(1,iter+1)
plt.plot(epochs, train_accs, 'g', label='Training Accuracy')
plt.plot(epochs, validation_accs, 'b', label='Validation Accuracy')
plt.title('Training Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Training')
plt.legend()
plt.show()  


epochs = range(1,iter+1)
plt.plot(epochs, train_losses, 'g', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()  

#%% Finding and Plotting Test Performance Metrics

# Calculate Accuracy         
correct_2 = 0
correct_3 = 0
total = 0
# Iterate through test dataset
for i, data in enumerate(test_loader):
    m = np.shape(data)[1]
    label = data[:,m-1]
    label = label.to(dtype=torch.int64)
    
    data = data[:,:m-1]
    data = np.reshape(data,(batch_size, 1, -1))
    images, labels = Variable(data), Variable(label)
    # Resize images
    images = images.view(-1, seq_dim, input_dim).requires_grad_()

    # Forward pass only to get logits/output
    outputs = model(images.to(device)).to(device)

    # Get predictions from the maximum value
    _, predicted = torch.topk(outputs.data, 3)

    # Total number of labels
    total += labels.size(0)

    # Total correct predictions
    correct_2 += (predicted[:,0].to('cpu') == labels).sum() + (predicted[:,1].to('cpu') == labels).sum()  
    correct_3 += (predicted[:,0].to('cpu') == labels).sum() + (predicted[:,1].to('cpu') == labels).sum()  + (predicted[:,2].to('cpu') == labels).sum() 

accuracy_top_2 = 100 * correct_2 / total
accuracy_top_3 = 100 * correct_3 / total

report_performance(predicted[:,0].to('cpu'), labels)

print("Top 2 prediction accuracy: %.1f%%" %(accuracy_top_2.item()))
print("Top 3 prediction accuracy: %.1f%%" %(accuracy_top_3.item()))