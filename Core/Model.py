import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import test_data_class
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
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def hidden_cell(self):
        h_0,c_0 = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size), 
        torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        return h_0,c_0 
    
    def forward(self, t):
        t, (h_n,c_n) = self.lstm(t, (self.hidden_cell()))
        t = self.linear(h_n[-1])
        return t[0]

if __name__ == "__main__":
    # Model
    model = LSTM(input_size = 14, hidden_size = 128 ,
                num_layers = 2, output_size = 1, batch_size = 1)
    learning_rate = 0.001
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 60
    
    

    # Training
    train = test_data_class.trainloader
    for epoch in range(epochs):
        for i, (seq, labels) in enumerate(train):
            optimizer.zero_grad()
            output = model(seq)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            
        if (epoch == 1):
            print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))

        #Evaluation
        test = test_data_class.testloader
        model.eval()
        predictions = []
        print("hello")
        with torch.no_grad():
            for i, (seq, labels) in enumerate(test):
                print("for")
                output = model(seq)
                print(output)
                predictions.append(output.item())
                print(predictions)
        predictions = np.add(np.multiply(predictions,test_data_class.df_std), test_data_class.df_mean)
        real = test_data_class.check
        print("RMS: " + str(ERRORS.RMS(predictions, real)) + "\n"
            "MAPE: " + str(ERRORS.MAPE(predictions, real)) + "\n"
            "MAE: " + str(ERRORS.MAE(predictions, real)) + "\n"
            "R: " + str(ERRORS.R(predictions, real)))    
        
        plt.plot(predictions)
        plt.plot(real)
        plt.show()



    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   













