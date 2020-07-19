# -*- coding: utf-8 -*-
"""
@author: Gabriel Gatti Lima
e-mail: gabriel.gatti.lima@usp.br
"""
import matplotlib as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def normalize(dataframe):
    """========================================================================
    Normalize a given Dataframe

    Input   ->  Denormalized Dataframe

    Outputs ->  1. Normalized dataframe, tuple containing
                2. tuple (average per feature, standart deviation per feature)
    ========================================================================"""
    media_ = dataframe.mean()
    sd_ = dataframe.std()
    dataframe_norm = (dataframe - media_)/sd_

    return [dataframe_norm, (media_, sd_)]


def denormalize(dataframe, stats):
    """========================================================================
    Denormalize a given Dataframe

    Inputs  ->  1. Normalized dataframe, tuple containing
                2. tuple (average per feature, standart deviation per feature)

    Output  ->  Denormalized Dataframe
    ========================================================================"""
    media_ = stats[0]
    sd_ = stats[1]
    return dataframe*sd_ + media_


def model_prediction(model, valid_ds, valid_target, stats_out,
                     denorm=False):
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
    prediction = model.predict(valid_ds)[:, -1]
    x = valid_target.shape[0]
    df_pred = pd.DataFrame([prediction, valid_target.reshape(x)])
    df_pred = df_pred.transpose()
    df_pred.columns = ['Predicted', 'Actual']

    if denorm:
        assert stats_out is not None
        media_ = stats_out[0]
        sd_ = stats_out[1]
        df_pred = denormalize(df_pred, (media_, sd_))

    df_pred['\u0394 Predicted'] = df_pred['Predicted'] - df_pred['Actual']

    return df_pred


def model_pipeline(dataframe, output_column, days_in_the_future=1, n_layers=3,
                   train_perc=(0.75, 0.15, 0.10), batch_size=32,
                   sampling_rate=1, length=256, epochs=1000, patience_=35,
                   learning_rate=10**-4):
    """========================================================================
    Create a pipeline wich consist of:
        1. Create datasets according to the hyperparameters
        2. Create a model according to the hyperparameters
        3. Compile the model
        4. Create the call-backs
        5. Train the model according to early-stop callback
        6. Load and return the model that had the lowest loss among all the
        epochs

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

    model_name = 'APP_LR= ' + str(learning_rate) + '#Layers=' + str(n_layers) \
        + ', #LG=' + str(length) + ', #BS=' + str(batch_size) + '.h5'

    # Creating Dataset---------------------------------------------------------
    [train_sets,
     dev_sets,
     valid_sets] = generate_dataset(dataframe, output_column,
                                    days_in_the_future=days_in_the_future,
                                    train_perc=train_perc,
                                    batch_size=batch_size,
                                    sampling_rate=sampling_rate,
                                    length=length)

    # Create Model-------------------------------------------------------------
    model = create_LSTM_model(len(dataframe.columns), length=length,
                              days_in_the_future=days_in_the_future,
                              drop_out=0.2, n_layers=n_layers)

    # Compilando---------------------------------------------------------------
    model.compile(loss=tf.keras.losses.Huber(), optimizer='adam',
                  metrics=["mape", "mae",
                           tf.keras.metrics.RootMeanSquaredError()])

    # Definindo Call-Backs-----------------------------------------------------
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                                                          save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience_)

    # Training the neural network----------------------------------------------
    history = model.fit(train_sets[0], epochs=epochs,
                        validation_data=dev_sets[0], steps_per_epoch=50,
                        callbacks=[early_stopping, model_checkpoint])

    # Loading the best model---------------------------------------------------
    model = tf.keras.models.load_model(model_name)

    return model, history, valid_sets


def create_LSTM_model(n_var_input, length=128, days_in_the_future=1,
                      drop_out=0.2, n_layers=3):
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
    model.add(layers.InputLayer(input_shape=(length, n_var_input)))

    # Group of layers----------------------------------------------------------
    for i in range(1, int(n_layers-1)):
        model.add(layers.LSTM(max((length+1)//2**i, days_in_the_future),
                              activation='sigmoid', return_sequences=True))

        model.add(layers.TimeDistributed(layers.Dense(
            (length+1)//max(2**(i+1), days_in_the_future),
            activation='sigmoid')))

        model.add(layers.Dropout(0.2))

    # Output group of layers---------------------------------------------------
    model.add(layers.LSTM(max((length+1)//2**n_layers, days_in_the_future),
              activation='sigmoid', return_sequences=False))

    model.add(layers.Dense(days_in_the_future, activation='sigmoid'))

    return model


def generate_dataset(dataframe, output_column, days_in_the_future=1,
                     train_perc=(0.7, 0.2, 0.1), batch_size=32,
                     sampling_rate=1, length=256):
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

    total_perc = train_perc[0] + train_perc[1] + train_perc[2]
    print(total_perc)
    assert (total_perc >= 0.999), "train_perc should sum up to 1"

    data = dataframe.values[:-days_in_the_future]
    target = np.array([dataframe[output_column].values[days_in_the_future:]])
    target = target.reshape(
        dataframe[output_column].values[:-days_in_the_future].shape[0], 1)
    train_split = int(len(dataframe.index)*train_perc[0])
    dev_split = int(len(dataframe.index)*(train_perc[0] + train_perc[1]))

    train_target = target[days_in_the_future:train_split]
    dev_target = target[length+train_split:length+dev_split]
    valid_target = target[length+dev_split:]

    train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data, target, length=length, sampling_rate=sampling_rate,
        batch_size=batch_size, end_index=train_split, shuffle=True)

    valid_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data, target, length=length, sampling_rate=sampling_rate,
        batch_size=batch_size, start_index=dev_split, shuffle=False)

    dev_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data, target, length=length, sampling_rate=sampling_rate,
        batch_size=batch_size, start_index=train_split,
        end_index=dev_split, shuffle=False)

    return [(train_gen, train_target),
            (dev_gen, dev_target),
            (valid_gen, valid_target)]


def naive_forecast(serie, days_in_the_future, stats_out, denorm=True):
    """========================================================================
    Create a naive forecast - Benchmark we are going use to evaluate our model

    Inputs  ->  1. serie:                Time series
                2. days_in_the_future:   Lenght of the prediction
                3. stats_out:            tuple(average of 1 feature,
                                               standart deviation of 1 feature)
                4. denormalize:          If we are going to denormalize or not

    Output  ->  Dataframe containing the naive forecast (Prediction Benchmark)
    ========================================================================"""

    actual = serie[days_in_the_future:]
    naive = serie[:-days_in_the_future]

    df_nv = pd.DataFrame([naive, actual], index=['Naive Forecast', 'Actual'])
    df_nv = df_nv.transpose()

    if denorm:
        assert stats_out is not None
        media_ = stats_out[0]
        sd_ = stats_out[1]
        df_nv = denormalize(df_nv, (media_, sd_))

    return df_nv


def show_predictions(model, valid_ds, valid_tgt, stats_out, denorm=False,
                     output_column=None, length=32, days_in_the_future=1):
    """========================================================================
    Show example of predictions, acutal values and the data used for prediciton

    Inputs  ->  1. serie:                Time series
                2. days_in_the_future:   Lenght of the prediction
                3. stats_out:            tuple(average of 1 feature,
                                               standart deviation of 1 feature)
                4. denormalize:          If we are going to denormalize or not

    Output  ->  Chart with 12 subplots
    ========================================================================"""

    prediction = model.predict(valid_ds)[:, -1]
    validation_target_1 = valid_tgt.reshape(valid_tgt.shape[0])

    if denorm:
        assert stats_out is not None
        media_ = stats_out[0]
        sd_ = stats_out[1]
        prediction = prediction*sd_ + media_
        validation_target_1 = validation_target_1*sd_ + media_

    plt.figure(figsize=(20, 16))
    for i in range(1, 13):
        initial_time = np.random.randint(0, len(prediction)-length)
        prediction_1 = prediction[initial_time+length]
        actual = validation_target_1[length]
        data_used_for_predict = validation_target_1[
            initial_time:initial_time+length]
        x = range(-(length+1), 0)

        plt.subplot(4, 3, i)
        sns.lineplot(x=x[:-1], y=data_used_for_predict, color='darkgreen')
        sns.scatterplot(x=[x[-1], x[-1]], y=[prediction_1, actual],
                        style=["predição", "real"], hue=["predição", "real"],
                        s=80)

    plt.show()
