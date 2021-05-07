import pandas as pd
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(1, alsologtostdout=False)

from pipeline import Training_Process
from lstm_model import LSTM_Model
from tensorflow.python.keras.utils.data_utils import Sequence
import datetime

config_dict={
    'HYPERPARAMETERS' : {
        'n_layers': (1, 4),
        'drop_out': (0, 0,5),
        'batch_size': [32, 64, 128, 256, 512],
        'length': (1, 49),
        'learning_rate': (-4, -1)
    },
    'DATA':{
        'FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/Unified Pickles',
        'INPUT_DF_NAME': 'Hourly_Complete_TOTAL.pkl',
    },
    'PARAMS': {
        'random_searchs': 2,
        'output_column': ['PM 2.5', 'Sulfur Dioxide (p.p.b)'],
        'patience': 1,
        'sampling_rate': 1,
        'days_in_future': 1,
        'division_perc': (0.6, 0.2, 0.2),
        'epochs': 2,
    },
}

def randomize_hyperparameter_tuning(hyper_dict:dict) -> dict:#(self, dataframe, output_column, hyper_dict, days_in_the_future=1, patience_=50, train_perc=(0.75, 0.15, 0.10), sampling_rate=1, epochs=1000, random_searchs=30):
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

    #Definindo os parametros e hyperparametros
    settings = config_dict['PARAMS']
    hyperparam_list = [randomize_hyperparameter_tuning(config_dict['HYPERPARAMETERS']) for i in range(0,config_dict['PARAMS']['random_searchs'])]

    lista_de_resultados = []
    #Criando os processo de trainamento
    for hyperparam_set in hyperparam_list:
        settings.update(hyperparam_set)

        # Train the model
        hyper_trained = Training_Process(df_input, **settings)
        
        settings['EVALUATION']: hyper_trained.evaluation
        settings['MODEL']: hyper_trained.model
        settings['HISTORY']: hyper_trained.history
        settings['MODEL_NAME']: hyper_trained.model_name
        
        lista_de_resultados.append(settings)
        
    return lista_de_resultados


pd.DataFrame(main(config_dict)).to_csv('/home/gabriel-gatti/Documents/air_pollution_forecast/resultado.csv')

