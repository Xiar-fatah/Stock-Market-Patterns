# Stock Market Prediction
The stock market is a chaotic and volatile statistical database.  It is therefore challenging toanalyze and predict.  In this project ”Predicting patterns in financial markets” we propose amachine learning-based approach in combination with feature extraction methods for stockmarket  prediction.   This  is  important  in  analysis  for  time  series  in  general  e.g.   speechrecognition.
## Model
### class LSTM
Creates the class for the model
### def __init__
The input for the nn.LSTM
* input_size: The number of expected features in the input, in our case it is opening price, closing price,...  and etc.
* hidden_size: The number of features in the hidden state "h".
* num_layers: Number of recurrent layers. E.g., setting "num_layers=2" would mean stacking two LSTMs together to form a "stacked LSTM with the second LSTM taking in outputs of the first LSTM and computing the final results. However, the default is set to 1.

The input for the nn.Linear
* in_features: size of each input sample. __Rewrite this in the code.__
* output_size:  size of each output sample

Other values that are initalizes
* batch_size: Size of the input batch, used in def hidden_cell. __Confirm this is correct and write an explanation.__

### hidden_cell 
The input of nn.LSTM is input, (h_0, c_0). In this function we create the tensors (h_0, c_0) which by default are zeros. This is done by 

__Confirm this, due to this maybe it is better to try Tensorflow.__

_h_0,c_0 = (torch.zeros(1, self.batch_size, self.hidden_size), torch.zeros(1, self.batch_size, self.hidden_size))_
        
### def forward
Forwards propagates the inputs into the model, the input is reshaped according to the following figure and feed into the model and then the linear layer.

###



### Output
output of shape (seq_len, batch, num_directions * hidden_size): 
tensor containing the output features (h_t) from the last layer of the LSTM, for each t. 

h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.

c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len.
