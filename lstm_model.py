import tensorflow as tf
from tensorflow.keras import layers

class LSTM_Model():

    def __init__(self, n_var_input, length=128, days_in_the_future=1, drop_out=0.2, n_layers=3):
        self.n_var_input = n_var_input
        self.length = length 
        self.days_in_future = days_in_the_future
        self.drop_out = drop_out
        self.n_layers = n_layers
        self.model = self.create_LSTM_model()

    def create_LSTM_model(self):
        """========================================================================
        Create a model according to the hyperparameters

        Input   ->  01. n_var_input:        number of features used to predict
                    02. length:             Number of time samples
                    03. days_in_the_future: Lenght of the prediction
                    04. drop_out:           Drop_out coefficient (regularization)
                    05. n_layers:           Number o layers of the model

        Outputs ->  Model

        ========================================================================"""
        model = tf.keras.models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.length, self.n_var_input)))

        # Group of layers----------------------------------------------------------
        for i in range(1, int(self.n_layers-1)):
            model.add(layers.LSTM(max((self.length+1)//2**i, self.days_in_future), activation='tanh', return_sequences=True))

            model.add(layers.TimeDistributed(layers.Dense(
                (self.length+1)//max(2**(i+1), self.days_in_future),
                activation='tanh')))

            model.add(layers.Dropout(self.drop_out))

        # Output group of layers---------------------------------------------------
        model.add(layers.LSTM(max((self.length+1)//2**self.n_layers, self.days_in_future),
                activation='tanh', return_sequences=False))

        model.add(layers.Dense(self.days_in_future, activation='tanh'))

        return model