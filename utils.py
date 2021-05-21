# -*- coding: utf-8 -*-
"""
@author: Gabriel Gatti Lima
e-mail: gabriel.gatti.lima@usp.br
"""
from typing import Callable
import pandas as pd
import datetime
import numpy as np

list_lat = [33.4339,
            33.553056,
            33.6878,
            35.2401,
            38.22876,
            38.587525,
            38.921847,
            39.70595,
            39.810833,
            39.83562,
            40.46542,
            40.720989,
            41.3014,
            41.530011,
            41.606667,
            41.821342,
            41.841039,
            41.97677,
            42.302786,
            42.3295,
            42.86183,
            42.862531,
            43.05235,
            43.14618,
            43.466111,
            43.608056,
            44.52839]

list_long = [-82.0224,
            -86.815,
            -84.2905,
            -80.78568299999999,
            -85.65451999999999,
            -76.141006,
            -77.013178,
            -79.012,
            -86.11444399999999,
            -84.720524,
            -79.960757,
            -74.192892,
            -72.90287099999999,
            -90.587611,
            -87.304722,
            -73.297257,
            -71.36097,
            -91.68766,
            -83.10653,
            -71.0826,
            -71.878626,
            -71.38014,
            -76.05921,
            -77.54817,
            -88.621111,
            -72.982778,
            -72.86884]

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
        df_nv = denormalize(df_nv, stats_out)

    return df_nv


def normalize(dataframe):
    """========================================================================
    Normalize a given Dataframe

    Input   ->  Denormalized Dataframe

    Outputs ->  1. Normalized dataframe, tuple containing
                2. tuple (average per feature, standart deviation per feature)
    ========================================================================"""
    media = dataframe.mean()
    sd = dataframe.std()
    
    return [(dataframe - media)/sd, (media, sd)]


def denormalize(dataframe, stats):
    """========================================================================
    Denormalize a given Dataframe

    Inputs  ->  1. Normalized dataframe, tuple containing
                2. tuple (average per feature, standart deviation per feature)

    Output  ->  Denormalized Dataframe
    ========================================================================"""
    return dataframe*stats[1] + stats[0]

def time_it(func:Callable, init_msg:str='', end_msg:str='Elapsed Time {duracao}'):
    if init_msg:
        print(init_msg)
    ini_time = datetime.datetime.now()
    resultado_func = func()
    duracao = datetime.datetime.now() - ini_time
    print(end_msg.format(duracao=duracao))

    return (resultado_func, duracao)

format_str = '%Y-%m-%d%H:%M'
hour_base = datetime.datetime.strptime('2000-01-0100:00', format_str)

def dateStr_2_Hours(date_str:str):
    format_str = '%Y-%m-%d%H:%M'
    hour_base=datetime.datetime.strptime('2000-01-0100:00', format_str)
    return (datetime.datetime.strptime(date_str, format_str) - hour_base).total_seconds()//3600

def Hours_2_datetime(hours):
    format_str = '%Y-%m-%d%H:%M'
    hour_base=datetime.datetime.strptime('2000-01-0100:00', format_str)
    
    return hour_base + datetime.timedelta(seconds=hours*3600)

def randomize_hyperparameter_tuning(hyper_dict:dict) -> list:
    """========================================================================
    khasghjasbvd
    Inputs  ->  
    Output  -> 
    ========================================================================"""
    n_layers = np.random.randint(hyper_dict['n_layers'][0], hyper_dict['n_layers'][1])
    batch_size = np.random.choice(hyper_dict['batch_size'])
    length = np.random.randint(hyper_dict['length'][0], hyper_dict['length'][1])
    learning_rate = 10**np.random.uniform(hyper_dict['learning_rate'][0], hyper_dict['learning_rate'][1])
    drop_out = np.random.uniform(hyper_dict['drop_out'][0], hyper_dict['drop_out'][1])
    
    return { 'n_layers': int(n_layers), 'drop_out': float(drop_out), 'batch_size': int(batch_size), 'length': int(length), 'learning_rate': float(learning_rate)}
