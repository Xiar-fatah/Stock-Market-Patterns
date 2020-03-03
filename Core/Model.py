import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data
from sklearn.preprocessing import MinMaxScaler
import ERRORS


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,batch_size):
        super(LSTM, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size 
        self.batch_size = batch_size 
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def hidden_cell(self):
        h_0,c_0 = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size), 
        torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        return h_0,c_0 
    
    def forward(self, t):
        t = t.reshape(20,1,1)
        t, (h_n,c_n) = self.lstm(t, (self.hidden_cell()))
        t = self.linear(h_n[-1])
        return t

if __name__ == "__main__":
    # Model
    model = LSTM(input_size = 1, hidden_size = 128 ,
                num_layers = 2, output_size = 1, batch_size = 1)
    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 60
    
    csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Core/MSFT.csv'

    

    # Training
    train = data.trainloader
    for epoch in range(epochs):
        for i, (seq, labels) in enumerate(train):
            optimizer.zero_grad()
            output = model(seq)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))

        #Evaluation
        test = data.testloader
        model.eval()
        predictions = []
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (seq, labels) in enumerate(test):
                output = model(seq)
                predictions.append(output.item())
                
        predictions = np.array(predictions)
        predictions = predictions * data.std + data.mean
        predictions = predictions.flatten()
        predictions = predictions.tolist()
        real = data.real
        print("RMS: " + str(ERRORS.RMS(predictions, real)) + "\n"
            "MAPE: " + str(ERRORS.MAPE(predictions, real)) + "\n"
            "MAE: " + str(ERRORS.MAE(predictions, real)) + "\n"
            "R: " + str(ERRORS.R(predictions, real)))    
        
        plt.plot(predictions)
        plt.plot(real)
        plt.show()



    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   













