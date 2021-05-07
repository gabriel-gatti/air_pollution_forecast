import pandas as pd
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(1, alsologtostdout=False)

from pipeline import Training_Process
from lstm_model import LSTM_Model
from itertools import chain
config_dict={
    'HYPERPARAMETERS' : {
        'N_LAYERS': (1, 4),
        'DROP_OUT': (0, 0,5),
        'BATCH_SIZE': [32, 64, 128, 256, 512],
        'LENGTH': (1, 49),
        'LEARNING_RATE': (-4, -1)
    },
    'DATA':{
        'FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp',
        'INPUT_DF_NAME': 'hourly_inner_join.pkl',
    },
    'PARAMS': {
        'RANDOM_SEARCHS': 1,
        'OUTPUT_COLUMN': 'Temperature (ºF)',
        'PATIENCE': 1,
        'SAMPLING_RATE': 1,
        'DAYS_IN_FUTURE': 1,
        'DIVISION_PERC': (0.6, 0.2, 0.2),
        'EPOCHS': 2,
    },
}



def generate_dataset(raw_dataframe, output_column, days_in_future=1, division_perc=(0.7, 0.2, 0.1), batch_size=32, sampling_rate=1, length=256):
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
    
    raw_dataframe.sort_values(by=['Latitude', 'Longitude', 'Date GMT', 'Time GMT'], inplace=True)
    raw_dataframe = raw_dataframe.drop_duplicates(subset=['Latitude', 'Longitude', 'Date GMT', 'Time GMT'], keep='first')

    train_gen, train_target, dev_gen, dev_target, valid_gen, valid_target = [], [], [], [], [], []

    for _, df_coord in raw_dataframe.groupby(by=['Latitude', 'Longitude']):
         
        df_target = df_coord[output_column]
        df_data = df_coord.drop(columns=output_column)

        data = df_data.values[:-days_in_future] 
        target = np.array([df_target.values[days_in_future:]])
        target = target.reshape(
            df_coord[output_column].values[:-days_in_future].shape[0], 1)
        train_split = int(len(df_coord.index)*division_perc[0])
        dev_split = int(len(df_coord.index)*(sum(division_perc[0:2])))

        # create targets list
        train_target.append(target[days_in_future:train_split])
        dev_target.append(target[length+train_split:length+dev_split])
        valid_target.append(target[length+dev_split:])

        # create generators list
        train_gen.append(tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data, target, length=length, sampling_rate=sampling_rate,
            batch_size=batch_size, end_index=train_split, shuffle=True))

        valid_gen.append(tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data, target, length=length, sampling_rate=sampling_rate,
            batch_size=batch_size, start_index=dev_split, shuffle=False))

        dev_gen.append(tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data, target, length=length, sampling_rate=sampling_rate,
            batch_size=batch_size, start_index=train_split,
            end_index=dev_split, shuffle=False))

    return train_gen, train_target, dev_gen, dev_target, valid_gen, valid_target


def main():
    #Definindo o DataFrame Base 
    df_input = pd.read_pickle('/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/hourly_inner_join.pkl')#mock_raw_df.pkl')

    return #generate_dataset(df_input, 'Temperature (ºF)')
    
#train_gen, train_target, dev_gen, dev_target, valid_gen, valid_target = mai
print('DONE !!!')



def time_series_generator():
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np
    data = np.array([[i] for i in range(14)])
    targets = np.array([[i] for i in range(14)])
    data_gen = TimeseriesGenerator(data, targets,
                                length=10, sampling_rate=2,
                                batch_size=2)
    assert len(data_gen) == 2
    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                        np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y,
                        np.array([[10], [11]]))
    return data, targets, data_gen

data, targets, data_gen = time_series_generator()

print('OK')