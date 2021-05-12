# -*- coding: utf-8 -*-
"""
@author: Gabriel Gatti Lima
e-mail: gabriel.gatti.lima@usp.br
"""

import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

