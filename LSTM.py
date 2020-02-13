import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data
import ERRORS
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,batch_size):
        super(LSTM, self).__init__()
        """
        
        input_size: The number of expected features in the input "x", in our case it is 1,
        the opening price.
        
        hidden_size: The number of features in the hidden state "h"
        
        num_layers: Number of recurrent layers. E.g., setting "num_layers=2"
            would mean stacking two LSTMs together to form a "stacked LSTM",
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
            
        more explanation
            
        """
        self.input_size = input_size #1 
        self.hidden_size = hidden_size #trial and error
        self.num_layers = num_layers #is a number between 1 and 3
        self.output_size = output_size #1
        self.batch_size = batch_size #60
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def hidden_cell(self):
        """
        h_0 of shape (num_layers * num_directions, batch, hidden_size):
        tensor containing the initial hidden state for each element in the batch.
        If the LSTM is bidirectional, num_directions should be 2, else it should be 1.

        c_0 of shape (num_layers * num_directions, batch, hidden_size): 
        tensor containing the initial cell state for each element in the batch.
        
        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        
        In essence this function creates two empty tensors to pass into the modell.
        """
        h_0,c_0 = (torch.zeros(1,self.batch_size,self.hidden_size), 
        torch.zeros(1,self.batch_size,self.hidden_size))
        return h_0,c_0 
    def forward(self, t):
        """
        output of shape (seq_len, batch, num_directions * hidden_size): 
        tensor containing the output features (h_t) from the last layer of the LSTM, for each t. 
        
        h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.
                
        c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len.
        """
        #Reshape the input tensor to satisfy (seq_len, batch, input_size) according to the docs
        t = t.view(len(t),1, -1)
        t, (h_n,c_n) = self.lstm(t, (self.hidden_cell()))
        t = t.view(-1, self.hidden_size)
        t = self.linear(t)
        #Why t[-1]? Read an article that said so, however there should exist a more cleaner way to do this
        t = t[-1]

        

        return t


if __name__ == "__main__":
    
    model = LSTM(input_size = 1, hidden_size = 100,
                 num_layers = 1, output_size = 1, batch_size = 1)
    learning_rate = 0.001

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 1

    for epoch in range(epochs):
        for seq, labels in data.train_data:
            optimizer.zero_grad()

            output = model(seq)
            loss = loss_function(output, torch.Tensor([labels]))
            loss.backward()
            optimizer.step()

        ##Add: Print loss function for given epoch.

    model.eval()
    predictions = []
    with torch.no_grad():
        correct = 0
        total = 0
        for seq, labels in data.test_data:
            output = model(seq)
            predictions.append(output.item())
    #Convert prediction from torch tensor to numpy array
    predictions = np.array(predictions)
    #Inverse the normalization
    predictions = data.sc.inverse_transform((predictions).reshape(-1, 1))

    ##Add dates instead of numbers
    trained_val = np.linspace(0,len(data.dataset)-1,len(data.dataset))
    test_val = np.linspace(1935,2034,100)

    plt.figure()
    plt.plot(trained_val,data.dataset,
             test_val, predictions)
    
    actaul_val = data.dataset[1935:]
    print("RMS: " + str(ERRORS.RMS(predictions, actaul_val)) + "\n"
          "MAPE: " + str(ERRORS.MAPE(predictions, actaul_val)) + "\n"
          "MAE: " + str(ERRORS.MAE(predictions, actaul_val)) + "\n"
          "R: " + str(ERRORS.R(predictions, actaul_val)))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    