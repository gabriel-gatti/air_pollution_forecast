import pandas as pd
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(1, alsologtostdout=False)

from pipeline import Training_Process
from lstm_model import LSTM_Model

config_dict={
    'HYPERPARAMETERS' : {
        'N_LAYERS': (1, 4),
        'DROP_OUT': (0, 0,5),
        'BATCH_SIZE': [32, 64, 128, 256, 512],
        'LENGTH': (1, 49),
        'LEARNING_RATE': (-4, -1)
    },
    'DATA':{
        'FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast',
        'INPUT_DF_NAME': 'hourly_inner_join.pkl',
    },
    'PARAMS': {
        'RANDOM_SEARCHS': 2,
        'OUTPUT_COLUMN': 'Temperature (ºF)',
        'PATIENCE': 1,
        'SAMPLING_RATE': 1,
        'DAYS_IN_FUTURE': 1,
        'DIVISION_PERC': (0.6, 0.2, 0.2),
        'EPOCHS': 1,
    },
}

def randomize_hyperparameter_tuning(hyper_dict:dict) -> dict:#(self, dataframe, output_column, hyper_dict, days_in_the_future=1, patience_=50, train_perc=(0.75, 0.15, 0.10), sampling_rate=1, epochs=1000, random_searchs=30):
            """========================================================================
            khasghjasbvd
            Inputs  ->  
float(
            Output  ->  
            ========================================================================"""
            n_layers = np.random.randint(hyper_dict['N_LAYERS'][0], hyper_dict['N_LAYERS'][1])
            batch_size = np.random.choice(hyper_dict['BATCH_SIZE'])
            length = np.random.randint(hyper_dict['LENGTH'][0], hyper_dict['LENGTH'][1])
            learning_rate = 10**np.random.uniform(hyper_dict['LEARNING_RATE'][0], hyper_dict['LEARNING_RATE'][1])
            drop_out = np.random.uniform(hyper_dict['DROP_OUT'][0], hyper_dict['DROP_OUT'][1])
            
            return { 'N_LAYERS': int(n_layers), 'DROP_OUT': float(drop_out), 'BATCH_SIZE': int(batch_size), 'LENGTH': int(length), 'LEARNING_RATE': float(learning_rate)}

def main(config_dict: dict):
    #Definindo o DataFrame Base 
    df_input = pd.read_pickle(f"{config_dict['DATA']['FOLDER_PATH']}/{config_dict['DATA']['INPUT_DF_NAME']}")

    #Definindo os parametros e hyperparametros
    settings = config_dict['PARAMS']
    hyperparam_list = [randomize_hyperparameter_tuning(config_dict['HYPERPARAMETERS']) for i in range(0,config_dict['PARAMS']['RANDOM_SEARCHS'])]

    lista_de_resultados = []
    #Criando os processo de trainamento
    for hyperparam_set in hyperparam_list:
        settings.update(hyperparam_set)

        # Train the model
        hyper_trained = Training_Process(
            df_input=df_input[['Latitude', 'Longitude', 'Temperature (ºF)', 'Relative Humidity (%)', 'Wind Speed - Resultant (knots)', 'Barometric pressure (Millibars)']],
            sampling_rate=settings['SAMPLING_RATE'],
            output_column=settings['OUTPUT_COLUMN'],
            division_perc=settings['DIVISION_PERC'],
            days_in_future=settings['DAYS_IN_FUTURE'],
            epochs=settings['EPOCHS'],
            patience=settings['PATIENCE'],
            batch_size=settings['BATCH_SIZE'],
            length=settings['LENGTH'],
            n_layers=settings['N_LAYERS'],
            drop_out=settings['DROP_OUT'],
            learning_rate=settings['LEARNING_RATE'])
        
        settings['EVALUATION']: hyper_trained.evaluation
        settings['MODEL']: hyper_trained.model
        settings['HISTORY']: hyper_trained.history
        settings['MODEL_NAME']: hyper_trained.model_name
        
        lista_de_resultados.append(settings)

    return lista_de_resultados


pd.DataFrame(main(config_dict)).to_csv('/home/gabriel-gatti/Documents/air_pollution_forecast/resultado.csv')

