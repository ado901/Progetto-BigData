import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

def neuralModel(X_train,y_train,X_val,y_val):
    
    #volevo provare tensorflow sulla gpu. puramente curiosità.
    with tf.device('/device:GPU:0'):
        model = models.Sequential()
        number= len(X_train.columns)
        #The first layer that you define is the input layer. This layer needs to know the input dimensions of your data.
        # Dense = fully connected layer (each neuron is fully connected to all neurons in the previous layer)
        model.add(layers.Dense(30, activation='relu',input_shape=(X_train.shape[1],)))
        #model.add(layers.Dense(int(29*0.75), activation='relu'))
        # Add one hidden layer (after the first layer, you don't need to specify the size of the input anymore)

        #1 neurone perchè classe binaria
        model.add(layers.Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[tf.keras.metrics.Precision(), tf.metrics.Recall()])
        # Fit the model to the training data and record events into aHistory object.
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,
        validation_split=0.2, verbose=0)
        # Model evaluation
        test_loss,test_pr, test_rec = model.evaluate(X_val,y_val)
        #ricavo f1
        f1=2/((1/test_pr)+ (1/test_rec))
        return(test_pr,test_rec,f1)
        #grafico andamento score
"""         plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(history.epoch,
        np.array(history.history['loss']),label='Train loss')
        plt.plot(history.epoch,
        np.array(history.history['val_loss']),label = 'Val loss')
        plt.legend()
        plt.show() """

