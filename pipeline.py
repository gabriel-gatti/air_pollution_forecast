import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
import datetime
from itertools import chain, cycle
from lstm_model import LSTM_Model
import matplotlib.pyplot as plt
import seaborn as sns
import utils

tf.autograph.set_verbosity(0, alsologtostdout=False)


class Training_Process():
    
    def __init__(self, df_input: pd.DataFrame, stats:tuple,  days_in_future: int, division_perc:tuple, batch_size: int, sampling_rate: int, length: int, output_column: list, n_layers:int, drop_out:float, epochs:int, patience:int, learning_rate:float, save_model_path:str, columns_to_drop:list,**kargs):
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
        self.stats = stats
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
        self.columns_to_drop=columns_to_drop
        self.model_name = self.create_model_name()
        print(self.model_name)
        self.model_folder = f'{save_model_path}{self.model_name}/' if save_model_path else self.model_name

        # create Frames
        #self.train_data, self.train_target, self.dev_data, self.dev_target, self.valid_data, self.valid_target = self.generate_dataset()
        self.train_data, self.dev_data, self.valid_data = self.generate_dataset()
        print('Datasets Generated !!!!')
        
        self.model, self.history, self.evaluation, self.valid_sets = self.model_pipeline()
        print('Training Process Finished')

    create_model_name = lambda self: f'MODEL_{datetime.datetime.now().strftime("%y%b%d-%Hh%Mm%Ss")}_#LearningRate={"{:.6f}".format(self.learning_rate)}_#Layers={self.n_layers}_#Length={self.length}_#BatchSize={self.batch_size}'
    
    def generate_dataset(self):
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

        train_gen, train_target, dev_gen, dev_target, valid_gen, valid_target = ([], [], [], [], [], [])
        self.row_count = 0
        datas, targets, splits = ([], [], [])
        for _, df_coord in self.raw_dataframe.groupby(by=['Latitude', 'Longitude']):
            # Remove possible torubles batching =====================================
            if df_coord.shape[0] > self.batch_size+1+self.length:
                # Define Target and Data based on the ouput column ==================
                df_target = df_coord[self.output_column]
                df_data = df_coord.drop(columns=self.columns_to_drop)

                # Create Data and Target Np.arrays ==================================
                data = df_data.values[:-self.days_in_future]
                datas.append(data) 

                target = np.array([df_target.values[self.days_in_future:]])
                target = target.reshape(
                    df_target.values[:-self.days_in_future].shape[0], len(self.output_column))
                targets.append(target)
                self.row_count += len(target)

                #defining split points ==============================================
                train_split = int(len(df_coord.index)*self.division_perc[0])
                dev_split = int(len(df_coord.index)*(sum(self.division_perc[0:2])))
                splits.append((train_split, dev_split))

                # create targets list ===============================================
                train_target.append(target[self.days_in_future:train_split])
                dev_target.append(target[self.length+train_split:self.length+dev_split])
                valid_target.append(target[self.length+dev_split:])

        zip_dados = lambda datas, targets, splits: zip(datas, targets, splits)

        # create generators of Timeseries ===========================================
        train_gen = [tf.keras.preprocessing.sequence.TimeseriesGenerator(
            dt, tgt, length=self.length, sampling_rate=self.sampling_rate,
            batch_size=self.batch_size, end_index=split[0], shuffle=True) for dt, tgt, split in zip_dados(datas, targets, splits)]

        valid_gen = [tf.keras.preprocessing.sequence.TimeseriesGenerator(
            dt, tgt, length=self.length, sampling_rate=self.sampling_rate,
            batch_size=self.batch_size, start_index=split[1], shuffle=False) for dt, tgt, split in zip_dados(datas, targets, splits)]

        dev_gen = [tf.keras.preprocessing.sequence.TimeseriesGenerator(
            dt, tgt, length=self.length, sampling_rate=self.sampling_rate,
            batch_size=self.batch_size, start_index=split[0],
            end_index=split[1], shuffle=False) for dt, tgt, split in zip_dados(datas, targets, splits)]

        #return train_gen, train_target, dev_gen, dev_target, valid_gen, valid_target
        return train_gen, dev_gen, valid_gen

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
        # Create Model-------------------------------------------------------------
        lstm_model = LSTM_Model(len(self.raw_dataframe.columns)-len(self.columns_to_drop), self.length, self.days_in_future, self.drop_out, self.n_layers)
        print('Model Created !!!!')

        # Definindo Call-Backs-----------------------------------------------------
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_folder+self.model_name+'.h5', save_best_only=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=self.patience_)
        print('Callbacks Defined !!!!')

        # Compilando---------------------------------------------------------------
        lstm_model.model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1),
                    metrics=["mape", "mae", tf.keras.metrics.RootMeanSquaredError()])
        print('Model Compiled !!!!')

        # Training the neural network----------------------------------------------
        history = lstm_model.model.fit(cycle(chain.from_iterable(self.train_data)), epochs=self.epochs,
                            validation_data=cycle(chain.from_iterable(self.dev_data)),
                            callbacks=[early_stopping, model_checkpoint],
                            use_multiprocessing=True, steps_per_epoch=(self.row_count//self.batch_size)*self.division_perc[0],
                            validation_steps=(self.row_count//self.batch_size)*self.division_perc[1])
        print('Model Trained !!!')

        # Loading the best model---------------------------------------------------
        #best_model = tf.keras.models.load_model(self.model_folder+self.model_name+'.h5')
        #print('Best Model Loaded !!!')
        best_model = lstm_model.model

        # Loading the best model---------------------------------------------------
        evaluation = best_model.evaluate(cycle(chain.from_iterable(self.valid_data)), steps=(self.row_count//self.batch_size)*self.division_perc[2])
        print('Model Evaluated !!!')

        return best_model, history, evaluation, (self.valid_data, self.valid_target)

    def save_predictions_overview(self):
        """========================================================================
        Return a dataframe containing the models predictions for the entire
        validation set asd

        Inputs -> model:              model
                valid_ds:           validation dataset
                valid_target:       validation target
                stats:              tuple (features avg, features sd)
                output_column:      feature we are trying to predict
                denormalize=False:  denormalze prediction or not?

        Output -> Prediction Dataframe containing the predicting and their
        respective actual values
        ========================================================================"""
        # Predict each Batch in the File===============
        predictions, actuals = ([], [])
        for coord in self.valid_data:
            for batch in coord:
                predictions.extend(self.model.predict(batch[0], batch_size=self.batch_size)[:, -1])
                actuals.extend(batch[1].reshape(batch[1].shape[0]))

        # Organize Dataframe =========================
        tpl_stats = tuple(map(lambda x: x[self.output_column].values[0], self.stats))
        df_pred = pd.DataFrame({'Predicted': predictions, 'Actual': actuals})
        df_pred = utils.denormalize(df_pred, tpl_stats)
        #df_pred['\u0394 Predicted'] = df_pred['Predicted'] - df_pred['Actual']

        #Save Plot Figure =============================
        max_v, min_v= (tpl_stats[0]+2*tpl_stats[1], tpl_stats[0]-2*tpl_stats[1])
        plt.figure()
        df_pred.plot(
            x='Predicted',
            y='Actual',
            figsize=(22, 10),
            title=f'Predictions do Modelo: {self.model_name}',
            xlabel='Batch item',
            ylabel='Normalized Prediction (non-dimensional)',
            kind='scatter',
            ylim=(min_v, max_v)
            ) #xlim=(min_v, max_v),

        plt.plot([min_v,max_v], [min_v,max_v], color='red')
        plt.savefig(self.model_folder+"Predictions_Batch_Chart.png")
        print(f'Predictions Overview Chart Created and Saved to {self.model_folder+"Predictions_Batch_Chart.png"}')

    def save_predictions_in_length(self):
        """========================================================================
        Show example of predictions, acutal values and the data used for prediciton

        Inputs  ->  1. serie:                Time series
                    2. days_in_the_future:   Lenght of the prediction
                    3. stats_out:            tuple(average of 1 feature,
                                                standart deviation of 1 feature)
                    4. denormalize:          If we are going to denormalize or not

        Output  ->  Chart with 12 subplots
        ========================================================================"""
        # Predict each Batch in the File =========================
        index_ouput = list(self.raw_dataframe.columns).index(self.output_column[0])
        x = range(-(self.length+1), 0)
        choosen_coords = np.random.choice(self.valid_data, 12)

        plt.figure(figsize=(20, 16))
        for i in range(1,len(choosen_coords)+1):
            batch =next(choosen_coords[i-1].__iter__())
            prediction = self.model.predict(np.array([batch[0][0]]), batch_size=1)[0][0]
            length = batch[0][0][:, index_ouput]
            actual = batch[1][0][0]

            plt.subplot(4, 3, i)
            sns.lineplot(x=x[:-1], y=length, color='darkgreen')
            sns.scatterplot(
                x=[x[-1], x[-1]],
                y=[prediction, actual],
                style=["Prediction", "Actual"],
                hue = ["Prediction", "Actual"],
                s=80)

        plt.savefig(self.model_folder+"Predictions_in_Length_Chart.png")
        print(f'Predictions in Length Chart Created and Saved to {self.model_folder+"Predictions_in_Length_Chart.png"}')
    
    def save_training_report(self):
        history_dev = {k: v for k, v in self.history.history.items() if k.find('val_')==-1}
        history_val = {k.replace('val_', ''): v for k, v in self.history.history.items() if k.find('val_')!=-1}
        history_unified = {k: {'train': history_dev[k], 'dev': history_val[k]} for k in history_dev.keys()}
        dict_df={key: pd.DataFrame(subhistory) for key, subhistory in history_unified.items()}

        plt.figure(figsize=(20, 16))
        fig, axes = plt.subplots(nrows=2, ncols=2)
        items = list(dict_df.items())
        items[0][1].plot(ax=axes[0,0], title=items[0][0])
        items[1][1].plot(ax=axes[0,1], title=items[1][0])
        items[2][1].plot(ax=axes[1,1], title=items[2][0])
        items[3][1].plot(ax=axes[1,0], title=items[3][0])
        
        plt.savefig(self.model_folder+"Training_Report_Chart.png")
        print(f'Training Report Chart Created and Saved to {self.model_folder+"Training_Report_Chart.png"}')
