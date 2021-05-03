import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
tf.autograph.set_verbosity(0, alsologtostdout=False)

import datetime

from lstm_model import LSTM_Model

class Training_Process():
    
    def __init__(self, df_input: pd.DataFrame,  days_in_future: int, division_perc:tuple, batch_size: int, sampling_rate: int, length: int, output_column: list, n_layers:int, drop_out:float, epochs:int, patience:int, learning_rate:float):
        try:   
            assert sum(division_perc) >= 0.999, f"division_perc should sum up to 1 | {division_perc}"
            assert map(lambda x: isinstance(x, str), output_column), f'output_column must be a list of str | {output_column}'
            assert isinstance(days_in_future, int) , f'days_in_future must be an Integer | {days_in_future}'
            assert isinstance(batch_size, int) , f'batch_size must be an Integer | {batch_size}'
            assert isinstance(sampling_rate, int) , f'sampling_rate must be an Integer | {sampling_rate}'
            assert isinstance(length, int) , f'length must be an Integer | {length}'
            assert isinstance(n_layers, int) , f'n_layers must be an Integer | {n_layers}'
            assert isinstance(df_input, pd.DataFrame) , f'df_input must be an DataFrame | {df_input}'
            assert isinstance(drop_out, int) , f'drop_out must be a Float | {drop_out}'
        except Exception as Err:
            print(f'Training_Process __init__ error | {Err}')

        # Define attributes
        self.drop_out = drop_out
        self.raw_dataframe = df_input
        self.n_layers = n_layers
        self.output_column = output_column
        self.days_in_future = days_in_future
        self.division_perc = division_perc
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.length = length
        self.epochs=epochs
        self.patience_=patience
        self.learning_rate=learning_rate
        self.model_name = f'MODEL_{datetime.datetime.now().strftime("%Y%b%d-%H%M%S")}_LR={str(self.learning_rate)}_#Layers={str(self.n_layers)}_#LG={str(self.length)}_#BS={str(self.batch_size)}.h5'
        print(self.model_name)

        self.model, self.history, self.evaluation, self.valid_sets = self.model_pipeline()
        

    def generate_dataset(self):#, dataframe, output_column, days_in_the_future=1, train_perc=(0.7, 0.2, 0.1), batch_size=32, sampling_rate=1, length=256):
        """========================================================================
        Create a pipeline wich consist of:
            1. Create datasets according to the hyperparameters

        Input   ->  01. dataframe:          Raw dataset we're going use to
                                            create the train, dev and valid sets
                    02. output_column:      Feature we are trying to predict
                    03. days_in_the_future: Lenght of the prediction
                    04. train_perc:         Tuple(train_perc, dev_perc, valid_perc)
                    05. batch_size:         Size of the batches
                    06. sampling_rate:      Used to create the datasets
                    07. length:             Number of time samples

        Outputs ->  List of tuples: [(train_gen, train_target),
                                    (dev_gen, dev_target),
                                    (valid_gen, valid_target)]
        ========================================================================"""        

        data = self.raw_dataframe.values[:-self.days_in_future]
        target = np.array([self.raw_dataframe[self.output_column].values[self.days_in_future:]])
        target = target.reshape(
            self.raw_dataframe[self.output_column].values[:-self.days_in_future].shape[0], 1)
        train_split = int(len(self.raw_dataframe.index)*self.division_perc[0])
        dev_split = int(len(self.raw_dataframe.index)*(sum(self.division_perc[0:2])))

        train_target = target[self.days_in_future:train_split]
        dev_target = target[self.length+train_split:self.length+dev_split]
        valid_target = target[self.length+dev_split:]

        train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data, target, length=self.length, sampling_rate=self.sampling_rate,
            batch_size=self.batch_size, end_index=train_split, shuffle=True)

        valid_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data, target, length=self.length, sampling_rate=self.sampling_rate,
            batch_size=self.batch_size, start_index=dev_split, shuffle=False)

        dev_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data, target, length=self.length, sampling_rate=self.sampling_rate,
            batch_size=self.batch_size, start_index=train_split,
            end_index=dev_split, shuffle=False)

        return train_gen, train_target, dev_gen, dev_target, valid_gen, valid_target

    def model_pipeline(self):
        """========================================================================
        Create a pipeline wich consist of:
            1. Create datasets according to the hyperparameters
            2. Create a model according to the hyperparameters
            3. Compile the model
            4. Create the call-backs
            5. Train the model according to early-stop callback
            6. Load and return the model that had the lowest loss among all the
            epochs
            7. Evaluate the model with the validation set

        Input   ->  01. dataframe:          Raw dataset we're going use to
                                            create the train, dev and valid sets
                    02. output_column:      Feature we are trying to predict
                    03. days_in_the_future: Lenght of the prediction
                    04. n_layers:           Number o layers of the model
                    05. train_perc:         Tuple(train_perc, dev_perc, valid_perc)
                    06. batch_size:         Size of the batches
                    07. sampling_rate:      Used to create the datasets
                    08. length:             Number of time samples
                    09. epochs:             Number of epochs (Very large since we
                                            are using early stopping)
                    10. patience:           Number of epochs without improving
                    11. learning_rate:      "Size" of gradient descent corrections

        Outputs ->  1. Trained Model:    Model that had the lowest loss among all
                                        the epochs
                    2. Training History: Metrics and loss at the end of each epoch
        ========================================================================"""
        # create Frames
        self.train_data, self.train_target, self.dev_data, self.dev_target, self.valid_data, self.valid_target = self.generate_dataset()

        # Create Model-------------------------------------------------------------
        lstm_model = LSTM_Model(len(self.raw_dataframe.columns), self.length, self.days_in_future, self.drop_out, self.n_layers)

        # Definindo Call-Backs-----------------------------------------------------
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_name, save_best_only=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.patience_)


        # Compilando---------------------------------------------------------------
        lstm_model.model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1),
                    metrics=["mape", "mae", tf.keras.metrics.RootMeanSquaredError()])

        # Training the neural network----------------------------------------------
        #print(type(self.epochs))
        #print(type(self.train_data))
        #print(type(self.dev_data))
        history = lstm_model.model.fit(self.train_data, epochs=self.epochs,
                            validation_data=self.dev_data,
                            callbacks=[early_stopping, model_checkpoint],
                            use_multiprocessing=True)

        # Loading the best model---------------------------------------------------
        best_model = tf.keras.models.load_model(self.model_name)

        # Loading the best model--------------------------------------------------_
        evaluation = best_model.evaluate(self.valid_data)

        return best_model, history, evaluation, (self.valid_data, self.valid_target)
