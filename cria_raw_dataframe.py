import pandas as pd
import os
import re
from functools import partial
from utils import time_it, dateStr_2_Hours, list_lat


def select_files_from_folder(origin_folder_path, regex_pattern: str) -> list:
    files_list = os.listdir(origin_folder_path)
    return list(filter(lambda file: re.search(regex_pattern, file), files_list))

@time_it
def inner_join_pickles(origin_folder:str, file_names:list=[], on_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT']):
    filenames = list(filter(lambda x:re.search('.*\.pkl', x),os.listdir(origin_folder))) if not file_names else file_names #: if None filename is passed uses all pickles in the folder
    df_temp = ''
    #Read File by File and Merge
    for file in filenames:
        fullname = f'{origin_folder}/{file}'
        if isinstance(df_temp, str): # First file to be read
            df_temp, _ = read_pickle_wrapper(fullname)
        else:
            df_temp, _ = pandas_merge_wrapper(df_temp, fullname, on_list)
    
    return df_temp

@time_it
def trata_dados(dataframe, index_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT'], lat:list=list_lat):
    df_temp = dataframe #deepcopy(dataframe)

    # Sort_Values and Drop_Duplicates
    df_temp.sort_values(by=index_list, inplace=True)
    df_temp.drop_duplicates(subset=index_list, keep='first', inplace=True)
    
    # Filter Coord
    df_temp = df_temp[df_temp['Latitude'].isin(lat)]

    # Format Date and Time
    df_temp['Ref_Hour'] = (df_temp['Date GMT'] + df_temp['Time GMT']).apply(dateStr_2_Hours)
    df_temp = df_temp.drop(columns=['Date GMT', 'Time GMT'])

    return df_temp

@time_it
def read_pickle_wrapper(path):
    return pd.read_pickle(path)

@time_it
def pandas_merge_wrapper(df_temp, fullname, on_list):
    df_temp2, _ = read_pickle_wrapper(fullname)
    return df_temp.merge(df_temp2, on=on_list, how="inner")

def load_dataframe(attr_path:str, merged_path:str, attr_list:list=['TEMP', 'RH_DP', 'SO2', 'WIND', 'PRESS', 'PM25'],
    on_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT'], index_list:list=['Latitude','Longitude', 'Date GMT', 'Time GMT'], use_cache=True):

    pos_name = f"Hourly_Merged_{'-'.join(sorted(attr_list))}.pkl"
    pos_fullname = f'{merged_path}/{pos_name}'
    
    # Search for cache
    if use_cache and os.path.exists(pos_fullname):
        return read_pickle_wrapper(pos_fullname)[0]

    # Create New File
    print(f'')
    df_temp, _ = inner_join_pickles(attr_path, list(map(lambda attr: f'hourly_{attr}_TOTAL.pkl', attr_list)), on_list,
    init_msg='Cache Not Found! Creating new File ' + pos_name + ' ...',
    end_msg='Tempo de criação do ' + pos_name + ': {duracao}')
    
    #Trata novo DataFrame
    df_temp, _ = trata_dados(df_temp, index_list, list_lat,
    init_msg='Tratando DataFrame ' + pos_name + ' ...',
    end_msg='Tempo de tratamento do ' + pos_name + ': {duracao}')
    
    # Save to Pickle
    print(f'Saving resulting DataFrame to : {pos_name}')
    df_temp.to_pickle(pos_fullname)

    return df_temp

def concat_attr(origin_path, destine_path):
    csvs = [f'{origin_path}/{x}' for x in select_files_from_folder(origin_path, '.*\.csv')]
    df_temp = None
    for csv in csvs:
        print(csv)
        if not isinstance(df_temp, pd.core.frame.DataFrame):
            df_temp = pd.read_csv(csv, usecols=['Latitude', 'Longitude', 'Date GMT', 'Time GMT', 'Sample Measurement'])
        else:
            df_temp = pd.concat([df_temp, pd.read_csv(csv, usecols=['Latitude', 'Longitude', 'Date GMT', 'Time GMT', 'Sample Measurement'])])
        
        print(df_temp)

    df_temp.to_pickle(destine_path)
    print(destine_path)


attr_list = ['hourly_44201']#['hourly_42101', 'hourly_42602', 'hourly_44201']
def concat_csvs(attr_list):
    origin = '/media/gabriel-gatti/HDD/Dados TCC/Attributes CSV/{attr_name}/'
    destino = '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/concat_per_attr/{attr_name}.pkl'

    for attr in attr_list:
        concat_attr(origin.format(attr_name=attr), destino.format(attr_name=attr))