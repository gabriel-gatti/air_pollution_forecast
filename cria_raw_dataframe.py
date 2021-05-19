import pandas as pd
import os
import re
from functools import partial
from utils import time_it, dateStr_2_Hours

config_cria_dataframe = {
    'attr_list': ['TEMP', 'RH_DP'], #['TEMP', 'RH_DP', 'SO2', 'WIND', 'PRESS', 'PM25'],
    'attr_path': '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/concat_per_attr',
    'merged_path': '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/merged',
    'on_list': ['Latitude','Longitude', 'Date GMT', 'Time GMT'],
    'use_cache': True,
    'index_list': ['Latitude','Longitude', 'Date GMT', 'Time GMT'],
}

def select_files_from_folder(origin_folder_path, regex_pattern: str) -> list:
    files_list = os.listdir(origin_folder_path)
    return list(filter(lambda file: re.search(regex_pattern, file), files_list))

def read_large_csv(caminho: str) -> pd.DataFrame:
    mylist = []
    return pd.concat([mylist.append(chunk) for chunk in pd.read_csv(caminho, low_memory=False, chunksize=20000)], axis=0)

def inner_join_pickles(origin_folder:str, file_names:list=[], on_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT']):
    filenames = list(filter(lambda x:re.search('.*\.pkl', x),os.listdir(origin_folder))) if not file_names else file_names #: if None filename is passed uses all pickles in the folder
    df_temp = ''
    #Read File by File and Merge
    for file in filenames:
        fullname = f'{origin_folder}/{file}'
        if isinstance(df_temp, str): # First file to be read
            df_temp, _ = time_it(partial(pd.read_pickle, fullname),
            init_msg='Lendo ' + file + ' ...',
            end_msg='Tempo de leitura do ' + file + ': {duracao}')
        else:
            df_temp, _ = time_it(partial(lambda df_temp, fullname, on_list: pd.merge(df_temp, pd.read_pickle(fullname), on=on_list, how="inner"), df_temp, fullname, on_list),
            init_msg='Lendo e executando InnerMerge ' + file + ' ...',
            end_msg='Tempo de leitura e Execução do InnerMerge do ' + file + ': {duracao}')
    
    return df_temp

def trata_dados(dataframe, index_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT']):
    df_temp = dataframe #deepcopy(dataframe)

    # Sort_Values and Drop_Duplicates
    df_temp.sort_values(by=index_list, inplace=True)
    df_temp.drop_duplicates(subset=index_list, keep='first', inplace=True)
    
    # Format Date and Time
    df_temp['Ref_Hour'] = (df_temp['Date GMT'] + df_temp['Time GMT']).apply(dateStr_2_Hours)
    df_temp = df_temp.drop(columns=['Date GMT', 'Time GMT'])

    return df_temp

def load_dataframe(attr_path:str, merged_path:str, attr_list:list=['TEMP', 'RH_DP', 'SO2', 'WIND', 'PRESS', 'PM25'],
    on_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT'], index_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT'], use_cache=True):

    pos_name = f"Hourly_Merged_{'-'.join(sorted(attr_list))}.pkl"
    pos_fullname = f'{merged_path}/{pos_name}'
    
    # Search for cache
    if use_cache and os.path.exists(pos_fullname):
        return time_it(partial(pd.read_pickle, pos_fullname),
        init_msg='Cache Found! Reading ' + pos_name + ' ...',
        end_msg='Tempo de Leitura do ' + pos_name + ': {duracao}')[0]

    # Create New File
    print(f'')
    df_temp, _ = time_it(partial(inner_join_pickles, attr_path, list(map(lambda attr: f'hourly_{attr}_TOTAL.pkl', attr_list)), on_list),
    init_msg='Cache Not Found! Creating new File ' + pos_name + ' ...',
    end_msg='Tempo de criação do ' + pos_name + ': {duracao}')
    
    #Trata novo DataFrame
    df_temp, _ = time_it(partial(trata_dados, df_temp, index_list),
    init_msg='Tratando DataFrame ' + pos_name + ' ...',
    end_msg='Tempo de tratamento do ' + pos_name + ': {duracao}')
    
    # Save to Pickle
    print(f'Saving resulting DataFrame to : {pos_name}')
    df_temp.to_pickle(pos_fullname)

    return df_temp
