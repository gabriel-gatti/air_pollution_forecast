
import matplotlib as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from utils import Utils as utl

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
            model.add(layers.LSTM(max((self.length+1)//2**i, self.days_in_future), activation='relu', return_sequences=True))

            model.add(layers.TimeDistributed(layers.Dense(
                (self.length+1)//max(2**(i+1), self.days_in_future),
                activation='relu')))

            model.add(layers.Dropout(self.drop_out))

        # Output group of layers---------------------------------------------------
        model.add(layers.LSTM(max((self.length+1)//2**self.n_layers, self.days_in_future),
                activation='relu', return_sequences=False))

        model.add(layers.Dense(self.days_in_future, activation='relu'))

        return model


    def model_prediction(self, valid_ds, valid_target):
        """========================================================================
        Return a dataframe containing the models predictions for the entire
        validation set

        Inputs -> model:              model
                valid_ds:           validation dataset
                valid_target:       validation target
                stats:              tuple (features avg, features sd)
                output_column:      feature we are trying to predict
                denormalize=False:  denormalze prediction or not?

        Output -> Prediction Dataframe containing the predicting and their
        respective actual values
        ========================================================================"""
        prediction = self.model.predict(valid_ds)[:, -1] #acho que é self.days_in_future
        x = valid_target.shape[0]
        df_pred = pd.DataFrame([prediction, valid_target.reshape(x)])
        df_pred = df_pred.transpose()
        df_pred.columns = ['Predicted', 'Actual']
        df_pred['\u0394 Predicted'] = df_pred['Predicted'] - df_pred['Actual']
        return df_pred

    def show_predictions(self, valid_ds, valid_tgt, output_column):
        """========================================================================
        Show example of predictions, acutal values and the data used for prediciton

        Inputs  ->  1. serie:                Time series
                    2. days_in_the_future:   Lenght of the prediction
                    3. stats_out:            tuple(average of 1 feature,
                                                standart deviation of 1 feature)
                    4. denormalize:          If we are going to denormalize or not

        Output  ->  Chart with 12 subplots
        ========================================================================"""

        prediction = self.model.predict(valid_ds)[:, -1]
        validation_target_1 = valid_tgt.reshape(valid_tgt.shape[0])

        #plt.figure(figsize=(20, 16))
        for i in range(1, 13):
            initial_time = np.random.randint(0, len(prediction)-self.length)
            prediction_1 = prediction[initial_time+self.length]
            actual = validation_target_1[self.length]
            data_used_for_predict = validation_target_1[
                initial_time:initial_time+self.length]
            x = range(-(self.length+1), 0)

            #plt.subplot(4, 3, i)
            sns.lineplot(x=x[:-1], y=data_used_for_predict, color='darkgreen')
            sns.scatterplot(x=[x[-1], x[-1]], y=[prediction_1, actual],
                            style=["predição", "real"], hue=["predição", "real"],
                            s=80)

        #plt.show()