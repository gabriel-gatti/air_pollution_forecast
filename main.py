import pandas as pd

from copy import deepcopy
import gc

from pipeline import Training_Process
import datetime
import json
import os
import utils
from cria_raw_dataframe import *
from feature_selection import run_feature_selection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_dict = {
    'HYPERPARAMETERS' : {
        'n_layers': (1, 3),
        'drop_out': (0, 0.5),
        'batch_size': [32, 64, 128, 256, 512],
        'length': (8, 49), #(8, 49),
        'learning_rate': (-6, -4)
    },
    'DATAFRAME': {
        'attr_list': ['PM25', 'PRESS', 'RH_DP', 'SO2', 'TEMP', 'WIND', 'CO', 'NO2', 'OZONE'], #['TEMP', 'RH_DP', 'SO2', 'WIND', 'PRESS', 'PM25'], # vira do Feature Selection
        'attr_path': '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/concat_per_attr',
        'merged_path': '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/merged',
        'on_list': ['Latitude','Longitude', 'Date GMT', 'Time GMT'],
        'use_cache': True,
        'index_list': ['Latitude','Longitude', 'Date GMT', 'Time GMT'],
    },
    'random_searchs': 8,
    'PARAMS': {
        'output_column': ['PM25'], #['PM 2.5', 'Sulfur Dioxide (p.p.b)'],
        'patience': 15,
        'sampling_rate': 1,
        'days_in_future': 1,
        'division_perc': (0.6, 0.2, 0.2),
        'epochs': 100,
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
    },
    'FEATURE_SELECTION': [
        ['RH_DP', 'TEMP', 'OZONE'],
        ['SO2', 'CO', 'NO2'],
        ['WIND', 'CO', 'NO2'],
        ['SO2', 'TEMP', 'CO'],
        ['TEMP', 'CO', 'NO2'],
        ['RH_DP', 'TEMP', 'NO2', 'OZONE'],
        ['SO2', 'WIND', 'CO', 'NO2'],
        ['WIND', 'CO', 'NO2', 'OZONE'],
        ['RH_DP', 'SO2', 'TEMP', 'CO'],
        ['RH_DP', 'TEMP', 'CO', 'NO2'],
        ['PRESS', 'RH_DP', 'TEMP', 'NO2', 'OZONE'],
        ['PRESS', 'SO2', 'WIND', 'CO', 'NO2'],
        ['SO2', 'WIND', 'CO', 'NO2', 'OZONE'],
        ['RH_DP', 'SO2', 'TEMP', 'CO', 'OZONE'],
        ['PRESS', 'RH_DP', 'TEMP', 'CO', 'NO2'],
        ['PRESS', 'RH_DP', 'TEMP', 'WIND', 'CO', 'NO2', 'OZONE'],
        ['PRESS', 'RH_DP', 'TEMP', 'WIND', 'CO', 'NO2', 'OZONE', 'SO2']
    ]
}

def main(config_dict: dict):
    #Definindo o DataFrame Base 

    """df_input = load_dataframe(**config_dict['DATAFRAME'])
    df_norm, stats = utils.normalize(df_input)
    stats = tuple(map(lambda x: x.drop(config_dict['PARAMS']['columns_to_drop']), stats))"""

    # Definindo Hyperparameters
    hyperparam_list = [utils.randomize_hyperparameter_tuning(config_dict['HYPERPARAMETERS']) for _ in range(0,config_dict['random_searchs'])]  # [{'n_layers': int(1), 'drop_out': float(0.25), 'batch_size': int(512), 'length': int(41), 'learning_rate': float(0.000015)}] #

    # Definindo variáveis relacionadas aos resultados
    lista_de_resultados = []
    if not os.path.exists(config_dict['PARAMS']['save_result_path']): os.mkdir(config_dict['PARAMS']['save_result_path'])
    if not os.path.exists(config_dict['PARAMS']['save_model_path']): os.mkdir(config_dict['PARAMS']['save_model_path'])

    #Criando os processo de trainamento ==========================================

    for fs_cols in config_dict['FEATURE_SELECTION']:
        # Feature Selection
        #fs_sets, fs_times = run_feature_selection(df_input=pd.read_pickle('/home/gabriel-gatti/Documents/Resultados TCC/Hourly_Merged_Final.pkl'), config=fs_config)
        settings_dataframe = deepcopy(config_dict['DATAFRAME'])
        settings_dataframe['attr_list'] = [fs_cols, *config_dict['PARAMS']['output_column']]
        #df_input = load_dataframe(**settings_dataframe)
        cols_used = ['Latitude', 'Longitude', 'Ref_Hour', *fs_cols, *config_dict['PARAMS']['output_column']]
        df_norm, stats = utils.normalize(pd.read_pickle('/home/gabriel-gatti/Documents/Resultados TCC/Hourly_Merged_Final.pkl')[cols_used])
        stats = tuple(map(lambda x: x.drop(config_dict['PARAMS']['columns_to_drop']), stats))

        for hyperparam_set in hyperparam_list:
            settings_hyper = deepcopy(config_dict['PARAMS'])
            settings_hyper.update(hyperparam_set)

            # Create model folder
            model_name = f'MODEL_{datetime.datetime.now().strftime("%y%b%d-%Hh%Mm%Ss")}_#LearningRate={"{:.6f}".format(settings_hyper["learning_rate"])}_#Layers={settings_hyper["n_layers"]}_#Length={settings_hyper["length"]}_#BatchSize={settings_hyper["batch_size"]}'
            model_folder = config_dict['PARAMS']['save_model_path'] + model_name + '/'
            if not os.path.exists(model_folder): os.mkdir(model_folder)

            # Train the model 
            hyper_trained = Training_Process(df_norm, stats, **settings_hyper)
            
            # Generate Reports ======================================================
            # Predictions
            hyper_trained.save_predictions_overview()
            hyper_trained.save_training_report()
            # hyper_trained.save_predictions_in_length()

            # Save Resultados a cada Modelo treinado para evitar perdas =============
            settings_hyper['Evaluation']                    = hyper_trained.evaluation
            settings_hyper['Attributes']                    = fs_cols
            settings_hyper['Num of Attributes']             = len(fs_cols)
            settings_hyper['Model Name']                    = hyper_trained.model_name
            #settings_hyper['Feature Selection Duration']    = fs_times[fs_name]
            settings_hyper['Training Duration']             = hyper_trained.training_duration
            settings_hyper.pop('save_model_path')
            settings_hyper.pop('save_result_path')
            settings_hyper.pop('columns_to_drop')

            df_resultado = pd.DataFrame([settings_hyper])
            df_resultado.to_csv(model_folder+'resultados')

            lista_de_resultados.append(deepcopy(settings_hyper))
            gc.collect()

        resultado_save_path = f"{config_dict['PARAMS']['save_result_path']}resultado_{datetime.datetime.now().strftime('%Y%b%d-%H%M%S')}_attr={str(fs_cols)}.csv"
        df_resultado_final = pd.DataFrame(lista_de_resultados)
        df_resultado_final.to_csv(resultado_save_path)

    return lista_de_resultados

if __name__ == '__main__':
    main(config_dict)


