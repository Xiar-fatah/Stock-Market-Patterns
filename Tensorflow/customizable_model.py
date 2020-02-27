import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

csv_path = 'https://raw.githubusercontent.com/Xiar-fatah/Stock-Market-Patterns/master/Tensorflow/fin.csv'
df = pd.read_csv(csv_path)

class LSTM(Model):
    def __init__(self):
        super(LSTM, self).__init__()


    def forward(self, x):
        
    
    return 




if __name__ == "__main__":
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    @tf.function
    def train_step(images, labels):
      with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
      train_loss(loss)
      train_accuracy(labels, predictions)
    @tf.function
    def test_step(images, labels):
      # training=False is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=False)
      t_loss = loss_object(labels, predictions)
    
      test_loss(t_loss)
      test_accuracy(labels, predictions)
      
      
      
            
      
      
            
      
      
            
      
      
            
    EPOCHS = 5

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
        train_loss.result(),
        train_accuracy.result()*100,
        test_loss.result(),
        test_accuracy.result()*100))
          
                
      
      
            
      
      
            
      
      
            
      
      
      
      