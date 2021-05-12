import pandas as pd
import numpy as np
from copy import deepcopy
from pipeline import Training_Process
import datetime
import json
import os
import utils
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_dict={
    'HYPERPARAMETERS' : {
        'n_layers': (1, 4),
        'drop_out': (0, 0,5),
        'batch_size': [32, 64, 128, 256, 512],
        'length': (8, 49),
        'learning_rate': (-4, -1)
    },
    'DATA':{
        'FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/Unified Pickles',
        'INPUT_DF_NAME': 'Hourly_Complete_PM25-SO2.pkl',
    },
    'PARAMS': {
        'random_searchs': 1,
        'output_column': ['PM 2.5'], #['PM 2.5', 'Sulfur Dioxide (p.p.b)'],
        'patience': 10,
        'sampling_rate': 1,
        'days_in_future': 1,
        'division_perc': (0.6, 0.2, 0.2),
        'epochs': 1,
        'save_model_path': '/home/gabriel-gatti/Documents/air_pollution_forecast/models/',
        'save_result_path': '/home/gabriel-gatti/Documents/air_pollution_forecast/resultados/',
        'columns_to_drop': ['Latitude', 'Longitude'],
    }
}

def randomize_hyperparameter_tuning(hyper_dict:dict) -> list:
            """========================================================================
            khasghjasbvd
            Inputs  ->  
float(
            Output  ->  
            ========================================================================"""
            n_layers = np.random.randint(hyper_dict['n_layers'][0], hyper_dict['n_layers'][1])
            batch_size = np.random.choice(hyper_dict['batch_size'])
            length = np.random.randint(hyper_dict['length'][0], hyper_dict['length'][1])
            learning_rate = 10**np.random.uniform(hyper_dict['learning_rate'][0], hyper_dict['learning_rate'][1])
            drop_out = np.random.uniform(hyper_dict['drop_out'][0], hyper_dict['drop_out'][1])
            
            return { 'n_layers': int(n_layers), 'drop_out': float(drop_out), 'batch_size': int(batch_size), 'length': int(length), 'learning_rate': float(learning_rate)}

def main(config_dict: dict):
    #Definindo o DataFrame Base 
    df_input = pd.read_pickle(f"{config_dict['DATA']['FOLDER_PATH']}/{config_dict['DATA']['INPUT_DF_NAME']}")
    df_norm, stats = utils.normalize(df_input)    
    stats = tuple(map(lambda x: x.drop(config_dict['PARAMS']['columns_to_drop']), stats))

    #Definindo os parametros e hyperparametros
    settings = config_dict['PARAMS']
    hyperparam_list = [randomize_hyperparameter_tuning(config_dict['HYPERPARAMETERS']) for i in range(0,config_dict['PARAMS']['random_searchs'])]

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

        # Save Resultados a cada Modelo treinado para evitar perdas =============
        lista_de_resultados.append(deepcopy(settings))
        df_resultado = pd.DataFrame(lista_de_resultados)
        df_resultado.to_csv(resultado_save_path)
        df_resultado.to_csv(model_folder+'resultados')

        print('BREAKPOINT')
    return lista_de_resultados

resultados = main(config_dict)

print('Done !!!!')