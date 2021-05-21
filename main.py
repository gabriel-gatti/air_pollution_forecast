import pandas as pd
import numpy as np
from copy import deepcopy
from pipeline import Training_Process
import datetime
import json
import os
import utils
from cria_raw_dataframe import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_dict = {
    'HYPERPARAMETERS' : {
        'n_layers': (1, 2),
        'drop_out': (0, 0,5),
        'batch_size': [32, 64, 128, 256, 512],
        'length': (8, 9), #(8, 49),
        'learning_rate': (-5, -2)
    },
    'DATAFRAME': {
        'attr_list': ['TEMP', 'RH_DP', 'WIND', 'PRESS', 'SO2', 'PM25'], #['TEMP', 'RH_DP', 'SO2', 'WIND', 'PRESS', 'PM25'], # vira do Feature Selection
        'attr_path': '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/concat_per_attr',
        'merged_path': '/home/gabriel-gatti/Documents/Resultados TCC', #'/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/merged',
        'on_list': ['Latitude','Longitude', 'Date GMT', 'Time GMT'],
        'use_cache': True,
        'index_list': ['Latitude','Longitude', 'Date GMT', 'Time GMT'],
    },
    'random_searchs': 1,
    'PARAMS': {
        'output_column': ['TEMP'], #['PM 2.5', 'Sulfur Dioxide (p.p.b)'],
        'patience': 15,
        'sampling_rate': 1,
        'days_in_future': 1,
        'division_perc': (0.6, 0.2, 0.2),
        'epochs': 50,
        'save_model_path': '/home/gabriel-gatti/Documents/Resultados TCC/models/',
        'save_result_path': '/home/gabriel-gatti/Documents/Resultados TCC/resultados/',
        'columns_to_drop': ['Latitude', 'Longitude', 'Ref_Hour'],
    },
    'TRANSLATION': {
        'TEMP': 'Temperature (ºF)',
        'RH_DP': 'Relative Humidity (%)',
        'SO2': 'Sulfur Dioxide (p.p.b)',
        'WIND': 'Wind Speed - Resultant (knots)',
        'PRESS': 'Barometric pressure (Millibars)',
        'PM25': ''
    }
}

def main(config_dict: dict):
    #Definindo o DataFrame Base 
    df_input = load_dataframe(**config_dict['DATAFRAME'])
    df_norm, stats = utils.normalize(df_input)
    stats = tuple(map(lambda x: x.drop(config_dict['PARAMS']['columns_to_drop']), stats))

    #Definindo os parametros e hyperparametros
    settings = config_dict['PARAMS']
    # [{'n_layers': int(1), 'drop_out': float(0.25), 'batch_size': int(512), 'length': int(41), 'learning_rate': float(0.000016)}]
    hyperparam_list = [{'n_layers': int(1), 'drop_out': float(0.25), 'batch_size': int(512), 'length': int(41), 'learning_rate': float(0.000016)}] #[utils.randomize_hyperparameter_tuning(config_dict['HYPERPARAMETERS']) for i in range(0,config_dict['random_searchs'])] 

    # Definindo variáveis relacionadas aos resultados
    lista_de_resultados = []
    if not os.path.exists(settings['save_result_path']): os.mkdir(settings['save_result_path'])
    if not os.path.exists(settings['save_model_path']): os.mkdir(settings['save_model_path'])
    resultado_save_path = f"{settings['save_result_path']}resultado_{datetime.datetime.now().strftime('%Y%b%d-%H%M%S')}.csv"

    #Criando os processo de trainamento ==========================================
    for hyperparam_set in hyperparam_list:
        settings.update(hyperparam_set)

        # Create model folder
        model_name = f'MODEL_{datetime.datetime.now().strftime("%y%b%d-%Hh%Mm%Ss")}_#LearningRate={"{:.6f}".format(settings["learning_rate"])}_#Layers={settings["n_layers"]}_#Length={settings["length"]}_#BatchSize={settings["batch_size"]}'
        model_folder = settings['save_model_path'] + model_name + '/'
        if not os.path.exists(model_folder): os.mkdir(model_folder)

        # Train the model 
        hyper_trained = Training_Process(df_norm, stats, **settings)

        settings['evaluation']= hyper_trained.evaluation
        settings['model']= hyper_trained.model.to_json()
        settings['history']= json.dumps(hyper_trained.history.history)
        settings['model_name']= hyper_trained.model_name
        
        # Generate Reports ======================================================
        # Predictions
        hyper_trained.save_predictions_overview()
        hyper_trained.save_predictions_in_length()
        hyper_trained.save_training_report()

        # Save Resultados a cada Modelo treinado para evitar perdas =============
        lista_de_resultados.append(deepcopy(settings))
        df_resultado = pd.DataFrame(lista_de_resultados)
        df_resultado.to_csv(resultado_save_path)
        df_resultado.to_csv(model_folder+'resultados')

        print('BREAKPOINT')
    return lista_de_resultados

resultados = main(config_dict)

#load_dataframe(**config_dict['DATAFRAME'])