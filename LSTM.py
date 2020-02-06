import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,batch_size):
        super(LSTM, self).__init__()
        """
        
        input_size: The number of expected features in the input `x`, in our case it is 1,
        the opening price.
        
        hidden_size: The number of features in the hidden state `h`
        
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
            
        """
        self.input_size = input_size
        self.hidden_size = hidden_size #trial and error
        self.num_layers = num_layers #is a number between 1 and 3
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, t):

        lstm_out, self.hidden_cell = self.lstm(t.view(len(t), self.batch_size,
                                                 -1,self.hidden_cell))
        predictions = self.linear(lstm_out.view(len(t), -1))
        return predictions
        


if __name__ == "__main__":
    
    model = LSTM(input_size = 1, hidden_size = 100,
                 num_layers = 1, output_size = 1, batch_size = len(data.X_train[0]))
    learning_rate = 0.001
    num_epoch = 100

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 150

    for i in range(epochs):
        for seq  in data.X_train[0]:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            y_pred = model(seq)
    
            single_loss = loss_function(y_pred, 1)
            single_loss.backward()
            optimizer.step()


