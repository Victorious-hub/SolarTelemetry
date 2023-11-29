import csv
import os.path
from datetime import datetime
from os import walk
from typing import List
import numpy as np
import pandas as pd
path_to_dataset = 'data'

def find_replace(csv_path, search_characters, replace_with):
    text = open(csv_path, "r")
    text = ''.join([i for i in text]).replace(
        search_characters, replace_with)
    file_write = open(csv_path, "w")
    file_write.writelines(text)
    file_write.close()


def get_list_panels():
    with open('panels_list.csv') as f:
        lines = f.readlines()
        return [f'Module {line}' for line in lines]
    
def check_file_exist(path: str):
    return os.path.exists(f'{path}.csv')

def hr_ts_func(ts):
    return ts.hour+2

def hr_ts_utc(ts):
    return ts.hour

def get_panels_dataset(path, panels):
    columns = ['timestamp', 'temperature', 'voltage', 'current', 'irradiation']
    x_train = []
    all_dataset = []

    for filename in panels:
        path_to_filename = os.path.join(path, filename)
        if not check_file_exist(path_to_filename):
            path_to_filename = os.path.join(path, filename.replace('.', '_'))
        if os.path.exists(f'{path_to_filename}.csv'):
            csv_path = f'{path_to_filename}.csv'
            search_characters = ';'
            replace_with = ','

            find_replace(csv_path, search_characters, replace_with)
            df = pd.read_csv(f'{path_to_filename}.csv')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['current'] = df['current'].astype(float)
            df['irradiation'] = df['irradiation'].astype(float)
            df['voltage'] = df['voltage'].astype(float)
            df['temperature'] = (df['temperature'].astype(float) + 273.2).round(decimals=2)

            df['time_hour_ts'] = df['timestamp'].apply(hr_ts_func)
            df['time_hour_utc_ts'] = df['timestamp'].apply(hr_ts_utc)
    
            panel_array = pd.DataFrame({
                'timestamp':df['timestamp'],
                'current':df['current'],
                'irradiation':df['irradiation'],
                'voltage':df['voltage'],
                'temperature':df['temperature'],
                'time_hour_ts':df['time_hour_ts'],
                'time_hour_utc_ts':df['time_hour_utc_ts']
            })  
            panel_array.to_csv('file.csv')

    return x_train, all_dataset


# get all modulesWithTemperatureDir folder datasets from data folder
def get_temperature():
    temperature_dir_name = 'modulesWithTemperatureDir'
    test_panel_names = [
        # 'Module 1.4_1'
        # 'Module 1.6_2'
        # 'Module 2.3_10'
        'Module 1.2_17',
        'Module 1.1_1',
        'Module 1.11_15'
    ]
    x_train = []
    x_test = []
    file = []
    for _, dirnames, filenames in walk(path_to_dataset):
        for dir_name in dirnames:
            print(dir_name, dirnames.index(dir_name) + 1, '/', len(dirnames))
            path = os.path.join(path_to_dataset, dir_name, temperature_dir_name)
            panels = get_list_panels()
            x_tr, file = get_panels_dataset(path, panels)
            print(str(len(x_tr)) + ' test records found')
            x_te, _ = get_panels_dataset(path, test_panel_names)
            print(str(len(x_te)) + ' test records found')

            x_train.extend(x_tr)
            x_test.extend(x_te)
        break
    print(file)
    return x_test, file
