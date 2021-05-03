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

class Utils:
    def naive_forecast(self, serie, days_in_the_future, stats_out, denorm=True):
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
            df_nv = self.denormalize(df_nv, (media_, sd_))

        return df_nv


    def normalize(self, dataframe):
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


    def denormalize(self, dataframe, stats):
        """========================================================================
        Denormalize a given Dataframe

        Inputs  ->  1. Normalized dataframe, tuple containing
                    2. tuple (average per feature, standart deviation per feature)

        Output  ->  Denormalized Dataframe
        ========================================================================"""
        media_ = stats[0]
        sd_ = stats[1]
        return dataframe*sd_ + media_
