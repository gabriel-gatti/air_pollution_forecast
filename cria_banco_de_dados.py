# %% Importando Bibliotecas
import pandas as pd
from datetime import datetime as dt
import os
import re
#  r"C:\Users\gabri\Documents\ESTAT\Projeto\Dados AIP\{}\{}_{}.csv".format(variavel, variavel, ano)

config = {
    'ORIGIN_FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast',
    'FILE_2_COL': { 
        'hourly_TEMP_TOTAL.pkl': 'Temperature (ºF)',
        'hourly_WIND_TOTAL.pkl': 'Wind Speed - Resultant (knots)',
        'hourly_RH_DP_TOTAL.pkl': 'Relative Humidity (%)',
        'hourly_PRESS_TOTAL.pkl': 'Barometric pressure (Millibars)'
    },
    'USE_COLS': ['Sample Measurement', 'Latitude', 'Longitude', 'Date GMT', 'Time GMT'],
    'DESTINATION_FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast'
}

class DataHandler:
    '''
    Description:

    params:

    methods:

    '''
    def __init__(self, config_dict: dict, extra_pars:dict={}):
        self.dataframes = {}
        self.origin_folder_path = config_dict.get('ORIGIN_FOLDER_PATH', 'Missing Folder Path')
        self.destination_folder_path = config_dict.get('DESTINATION_FOLDER_PATH', 'Missing Folder Path')

    def select_files_from_folder(self, regex_pattern: str) -> list:
        files_list = os.listdir(self.origin_folder_path)
        return list(filter(lambda file: re.search(regex_pattern, file), files_list))

    def read_large_csv(self, caminho: str) -> pd.DataFrame:
        mylist = []
        return pd.concat([mylist.append(chunk) for chunk in pd.read_csv(caminho, low_memory=False, chunksize=20000)], axis=0)
    
    def write_2_pkl(self, df_key_list:list=[]):
        df_key_list = df_key_list if len(df_key_list)>0 else list(self.dataframes.keys())
        for df_key in df_key_list:
            temp = self.dataframes.get(df_key, False)
            if temp:
                self.dataframes.get(df_key, False).to_pickle(self.read_large_csv(f'{self.destination_folder_path}/{df_key}'))
    
    def del_df_from_handler(self, df_key_list:list=[]):
        df_key_list = df_key_list if df_key_list else list(self.dataframes.keys())
        for df_key in df_key_list:
            self.dataframes.pop(df_key, None)

    def load_csv_files(self, file_names:list, save2memory:bool=False):
        file_names = file_names if isinstance(file_names, list) else [file_names]
        for file_name in file_names:
            print(f'Reading: ' + self.origin_folder_path + '/' + file_name)
            self.dataframes[file_name] = self.read_large_csv(f'{self.origin_folder_path}/{file_name}')
    
    def load_pkl_files(self, file_names:list):
        file_names = file_names if isinstance(file_names, list) else [file_names]
        for file_name in file_names:
            print(f'Reading: ' + self.origin_folder_path + '/' + file_name)
            self.dataframes[file_name] = pd.read_pickle(f'{self.origin_folder_path}/{file_name}')


# %%
def main():
    data_handler = DataHandler(config)
    
    data_handler.load_pkl_files(data_handler.select_files_from_folder('.*\.pkl')[0])

    #df.reset_index(inplace=True) for k, df in data_handler.dataframes.items()]

    return data_handler

df_data = main()
# %% Data Exploration
'''
comeco = dt.now()


for variavel, nome_variavel in dict_nome_arq.items():

    time_temp = dt.now()
    df_main = pd.read_csv(r"/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/{}/{}_{}.csv".format(variavel, variavel, 2000), low_memory=True, usecols=use_cols)
    print('Tempo de leitura do {}: {}'.format(2000, dt.now() - time_temp))

    semi_comeco = dt.now()
    for ano in range(2001, 2021):

        time_temp = dt.now()
        mylist = []
        for chunk in pd.read_csv(r"/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/{}/{}_{}.csv".format(variavel, variavel, ano), low_memory=False, usecols=use_cols, chunksize=20000):
            mylist.append(chunk)

        df_temp = pd.concat(mylist, axis=0)
        del mylist
        del chunk
        print('Tempo de leitura do {}_{}: {}'.format(variavel, ano, dt.now() - time_temp))

        time_temp = dt.now()
        df_main = pd.concat([df_main, df_temp])
        df_main
        print('Tempo de concat do {}_{}: {}'.format(variavel, ano, dt.now() - time_temp))

    del df_temp

    time_temp = dt.now()
    df_main.rename({'Sample Measurement': nome_variavel})
    df_main.to_csv(r"/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/{}_TOTAL_antes.csv".format(variavel))
    del df_main
    print('Tempo de escrita do CSV: {}'.format(dt.now() - time_temp))

    print('Tempo semi-TOTAL {}: {}'.format(variavel, dt.now() - semi_comeco))

print('Tempo TOTAL: {}'.format(dt.now() - comeco))

#%%
comeco = dt.now()

df_main = le_csv_grande(r"/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/hourly_PRESS_TOTAL_antes.csv")
df_main.rename({'Sample Measurement': 'Barometric pressure (Millibars)'})
df_main.set_index(['Latitude', 'Longitude','Date GMT', 'Time GMT'])

for variavel, nome_variavel in [
        ['hourly_TEMP', 'Temperature (ºF)'],
        ['hourly_WIND', 'Wind Speed - Resultant (knots)'],
        ['hourly_RH_DP', 'Relative Humidity (%)']]:
        #['hourly_PRESS', 'Barometric pressure (Millibars)']]:

    df_temp = le_csv_grande(r"/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/{}_TOTAL_antes.csv".format(variavel))
    df_temp.rename({'Sample Measurement': nome_variavel})
    df_temp.set_index(['Latitude', 'Longitude','Date GMT', 'Time GMT'])

    time_temp = dt.now()
    df_main = pd.concat([df_main, df_temp], axis=1)
    print('Tempo de concat do {}: {}'.format(variavel, dt.now() - time_temp))


df_main.to_csv(r"C:/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/UNIFICADO_2000-2020.csv")
print('Tempo TOTAL: {}'.format(dt.now() - comeco))

'''
#%%
'''
site_ = 3

df = df_temperatura
df_temporario = df[df['Site Num']==site_][['Date Local','Time Local','Sample Measurement']]
df_temporario['Date-Time'] = df_temporario['Date Local'] + ' _ ' + df_temporario['Time Local']
df_temporario = df_temporario[['Date-Time','Sample Measurement']].set_index('Date-Time')
df_temporario = df_temporario.groupby(df_temporario.index)[['Sample Measurement']].mean()
df_temperatura_1 = df_temporario.rename(columns = {"Sample Measurement" : "Temperature (F)"})

df = df_pressao
df_temporario = df[df['Site Num']==site_][['Date Local','Time Local','Sample Measurement']]
df_temporario['Date-Time'] = df_temporario['Date Local'] + ' _ ' + df_temporario['Time Local']
df_temporario = df_temporario[['Date-Time','Sample Measurement']].set_index('Date-Time')
df_temporario = df_temporario.groupby(df_temporario.index)[['Sample Measurement']].mean()
df_pressao_1 = df_temporario.rename(columns = {"Sample Measurement" : "Pressao"})

df = df_umidade
df_temporario = df[df['Site Num']==site_][['Date Local','Time Local','Sample Measurement']]
df_temporario['Date-Time'] = df_temporario['Date Local'] + ' _ ' + df_temporario['Time Local']
df_temporario = df_temporario[['Date-Time','Sample Measurement']].set_index('Date-Time')
df_temporario = df_temporario.groupby(df_temporario.index)[['Sample Measurement']].mean()
df_umidade_1 = df_temporario.rename(columns = {"Sample Measurement" : "Umidade"})

df = df_vento
df_temporario = df[df['Site Num']==site_][['Date Local','Time Local','Sample Measurement']]
df_temporario['Date-Time'] = df_temporario['Date Local'] + ' _ ' + df_temporario['Time Local']
df_temporario = df_temporario[['Date-Time','Sample Measurement']].set_index('Date-Time')
df_temporario = df_temporario.groupby(df_temporario.index)[['Sample Measurement']].mean()
df_vento_1 = df_temporario.rename(columns = {"Sample Measurement" : "Vento"})

df = df_44201
df_temporario = df[df['Site Num']==site_][['Date Local','Time Local','Sample Measurement']]
df_temporario['Date-Time'] = df_temporario['Date Local'] + ' _ ' + df_temporario['Time Local']
df_temporario = df_temporario[['Date-Time','Sample Measurement']].set_index('Date-Time')
df_temporario = df_temporario.groupby(df_temporario.index)[['Sample Measurement']].mean()
df_44201_1 = df_temporario.rename(columns = {"Sample Measurement" : "44201"})

df_result.plot(subplots=True, layout=(5,1), figsize=(22,22))
#%% TREATING DATA
dicionar = {'42101'	: 'Carbon monoxide',
            '42401' : 'Sulfur dioxide',
            '42402' : 'Hydrogen sulfide',
            '42600' : 'Reactive oxides of nitrogen (NOy)',
            '42601' : 'Nitric oxide (NO)',
            '42602' : 'Nitrogen dioxide (NO2)',
            '42603' : 'Oxides of nitrogen (NOx)',
            '42612' : 'NOy - NO',
            '43102' : 'Total NMOC (non-methane organic compound)',
            '44201' : 'Ozone'}


df_result = pd.read_csv("Dados AIP\Site_3_NY_Agrupado.csv", low_memory=False)
features = df_result.reset_index().interpolate(method= 'cubic').dropna(axis=0).set_index('Date-Time').drop(['index'], axis=1).rename(columns= dicionar)
features.plot(subplots=True, layout=(5,1), figsize=(8,6))
features_indexes = features.index
features_columns = features.columns
out_col = "Ozone (Parts per million)"
'''
