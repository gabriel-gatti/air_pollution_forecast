
import pandas as pd
from datetime import datetime as dt
import os
import re
#  r"C:\Users\gabri\Documents\ESTAT\Projeto\Dados AIP\{}\{}_{}.csv".format(variavel, variavel, ano)

config = {
    'ORIGIN_FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/Unified Pickles',
    'USE_COLS': ['Sample Measurement', 'Latitude', 'Longitude', 'Date GMT', 'Time GMT'],
    'DESTINATION_FOLDER_PATH': '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp'
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

    def load_csv_files(self, file_names:list):
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
import pandas as pd
from datetime import datetime as dt
import datetime
import os
import re

def select_files_from_folder(origin_folder_path, regex_pattern: str) -> list:
    files_list = os.listdir(origin_folder_path)
    return list(filter(lambda file: re.search(regex_pattern, file), files_list))

def read_large_csv(caminho: str) -> pd.DataFrame:
    mylist = []
    return pd.concat([mylist.append(chunk) for chunk in pd.read_csv(caminho, low_memory=False, chunksize=20000)], axis=0)

path = '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/Unified Pickles/merged'
df_temp = ''
for file in select_files_from_folder(path, '.*\.pkl'):
    filename = f'{path}/{file}'
    if isinstance(df_temp, str):
        print(f'Reading: {file}')
        time_temp = dt.now()
        df_temp = pd.read_pickle(filename)
        print(f'Tempo de leitura do {file}: {dt.now() - time_temp}')
    else:
        print(f'Reading and Inner Joining: {file}')
        df_temp = pd.merge(df_temp, pd.read_pickle(filename), on=['Latitude','Longitude', 'Date GMT', 'Time GMT'], how="inner")
        print(f'Tempo de leitura e concatenação do {file}: {dt.now() - time_temp}')

pos_name = path  + '/Hourly_Complete_TOTAL_PM25.pkl'
df_temp.sort_values(by=['Latitude','Longitude', 'Date GMT', 'Time GMT'], inplace=True)
df_temp.drop_duplicates(subset=['Latitude','Longitude', 'Date GMT', 'Time GMT'], keep='first', inplace=True)
# Format Date and Time
df_temp['Day'] = df_temp['Date GMT'].apply(lambda x: (datetime.date(*list(map(int,x.split('-')))) - datetime.date(int(x.split('-')[0]),1,1)).days)
df_temp['Hour'] = df_temp['Time GMT'].apply(lambda x: int(str(x[:2])))
df_temp = df_temp.drop(columns=['Date GMT', 'Time GMT'])

print('Saving Pickle to : Hourly_Complete_TOTAL_PM25.pkl')
df_temp.to_pickle(pos_name)


#%%
path = '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/Unified Pickles/merged/Hourly_Complete_TOTAL_PM25.pkl'



#%%
'''
dict_types = {
    'hourly_42401': ('Sulfur Dioxide', 'ppb'),
    'hourly_88101': ('PM 2.5', 'Microgrmas/Cubic Meter'),
}

path = '/media/gabriel-gatti/HDD/Users/gabri/Documents/ESTAT/Projeto/Dados AIP/'
dest_path = '/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/'

for key in dict_types.keys():
        for file in select_files_from_folder('/home/gabriel-gatti/Documents/air_pollution_forecast_BackUp/Unified Pickles', '.*\.pkl'):
        filename = path+key+'/'+file
        if isinstance(df_temp, str):
            print(f'Reading: {file}')
            time_temp = dt.now()
            df_temp = pd.read_pickle(filename)
            print(f'Tempo de leitura do {file}: {dt.now() - time_temp}')
        else:
            print(f'Reading and Concating: {file}')
            df_temp = pd.concat([df_temp, pd.read_csv(filename)
            print(f'Tempo de leitura do {file}: {dt.now() - time_temp}')

    pos_name = dest_path + key + '_TOTAL.pkl'
    print('Saving Pickle to : ' + pos_name)

    df_temp.rename(columns={'Sample Measurement': dict_types[key][0]})
    df_temp.to_pickle(pos_name)

'''
#%%
'''
df_input = df_input.apply(lambda x: datetime.datetime.strptime(' '.join([x['Date GMT'], x['Time GMT']]), '%Y-%m-%d %H:%M'), axis=1)
df_input = df_input.drop(columns=['Date GMT', 'Time GMT'])

df_input.to_pickle(f"{config_dict['DATA']['FOLDER_PATH']}/{config_dict['DATA']['INPUT_DF_NAME']}")
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
