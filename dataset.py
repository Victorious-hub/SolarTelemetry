import csv
import json
import os.path
from datetime import datetime
from os import walk

import numpy as np
from keras.src.datasets import mnist

path_to_dataset = 'data'


# mnist is just a tensorflow dataset with numeric data, ranging between 0 and 9 and specific channel 1(gray color)
def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # get images with 28x28 size and channel = 1
    return x_train, x_test


# get some int and turn it into the datetime
def get_time(timestamp: int) -> datetime:
    date = datetime.fromtimestamp(int(timestamp) / 1e3)  # It will return the datetime object.
    return date


# this code analyses some distinct folders in main and it's files inside
def get_list_panels():  # reads Panels_list.csv dataset, converting coma to dot
    with open('panels_list.csv') as f:
        lines = f.readlines()
        return [f'Module {line.rstrip().replace(",", ".")}' for line in lines]


# print(check_file_exist("Panels_list")) -> True
def check_file_exist(path: str):
    return os.path.exists(f'{path}.csv')


# .csv files from data-folder datasets(June_2019, July2019, August_2019, November_2019, October_2019, September_2019)
def get_panels_dataset(path, panels):
    structure = ['timestamp', 'temperature', 'voltage', 'current', 'irradiation']
    x_train = []
    all_dataset = []
    ald = []

    for filename in panels:
        path_to_filename = os.path.join(path, filename)
        if not check_file_exist(path_to_filename):
            path_to_filename = os.path.join(path, filename.replace('.', '_'))
        if os.path.exists(f'{path_to_filename}.csv'):
            with open(f'{path_to_filename}.csv') as f:
                reader = csv.DictReader(f, delimiter=";")
                counter = 0
                prev_day = 0
                wtf_image = []
                prev_irr = 0
                prev_curr = 0
                prev_index = -1
                for r in reader:
                    time = get_time(r['timestamp'])
                    current = float(r['current'])
                    irradiating = float(r['irradiation'])
                    voltage = float(r['voltage'])
                    temperature = float(r['temperature']) + 273.2
                    if prev_day != time.day:
                        counter = 0
                        prev_day = time.day
                        prev_irr = 0
                        prev_curr = 0
                        prev_index = -1

                        if wtf_image and 100 <= len(wtf_image) < 200:
                            i = len(wtf_image)
                            s = wtf_image[i - 1][0] + 1
                            while i < 200:
                                rx = [s, 0, 0, 0, 0]
                                s = s + 1
                                wtf_image.append(rx)
                                # ald.append(rx)
                                i = i + 1

                        while len(wtf_image) > 200:
                            if wtf_image and len(wtf_image) > 200:
                                gap = len(wtf_image) - 200
                                if gap > 200:
                                    gap = 200
                                mid = int(len(wtf_image) / 2)
                                wtf_image_tmp = []
                                ald_tmp = []
                                index = mid - (1 + 2 * (int((gap - 1) / 2)))
                                i = 0
                                while gap > 0:
                                    if i != index:
                                        wtf_image_tmp.append(wtf_image[i])
                                        ald_tmp.append(ald[i])
                                    else:
                                        index = index + 2
                                        gap = gap - 1
                                    i = i + 1
                                wtf_image = wtf_image_tmp
                                ald = ald_tmp
                        if wtf_image and len(wtf_image) == 200:
                            """
                            i = 74
                            while i<125:
                                wtf_image[i][2] = 0
                                i = i + 1
                            """
                            x_train.append(wtf_image)
                            all_dataset.extend(ald)
                        ald = []
                        wtf_image = []
                    """
                    if time.hour >= 8 and time.hour < 20:
                        counter += 1
                        val = [float(r[s]) for s in structure]
                        val[0] = float((time.hour - 8)*60 + time.minute)
                        wtf_image.append(val)
                        ald.append(r)
                    """
                    if 8 <= time.hour < 20 and 360 <= irradiating <= 1500:  # and current >=0 and current <=15 and temperature >= 25 and temperature <= 400
                        index = float((time.hour - 8) * 60 + time.minute)
                        if index == prev_index:
                            index = index + 0.5
                        deriv_curr = (current - prev_curr) / (
                                index - prev_index)  # derivative of current:     -0.2 – +0.2
                        deriv_irr = (irradiating - prev_irr) / (
                                index - prev_index)  # derivative of irradiation: -3.8 – +3.8
                        if prev_irr != irradiating:
                            if -3.8 <= deriv_irr <= 3.8:  # and deriv_curr >= -0.2 and deriv_curr <= 0.2
                                # print(r)
                                counter += 1
                                val = [float(r[s]) for s in structure]
                                val[0] = index
                                val[3] = val[3]
                                wtf_image.append(val)
                                ald.append(r)
                                prev_irr = irradiating
                                prev_curr = current
                                prev_index = index
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
    all_dataset = []
    for _, dirnames, filenames in walk(path_to_dataset):
        for dir_name in dirnames:
            print(dir_name, dirnames.index(dir_name) + 1, '/', len(dirnames))
            path = os.path.join(path_to_dataset, dir_name, temperature_dir_name)
            panels = get_list_panels()
            x_tr, a_dr = get_panels_dataset(path, panels)
            print(str(len(x_tr)) + ' test records found')
            x_te, _ = get_panels_dataset(path, test_panel_names)
            print(str(len(x_te)) + ' test records found')

            x_train.extend(x_tr)
            x_test.extend(x_te)
            all_dataset.extend(a_dr)
        break

    # structure = ['timestamp', 'temperature', 'voltage', 'current', 'irradiation']
    # maxs = {f'{s}_max': max(float(d[s]) for d in all_dataset) for s in structure}
    # mins = {f'{s}_min': min(float(d[s]) for d in all_dataset) for s in structure}
    # print('maxs:', maxs, 'mins:', mins)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (len(x_train), 200, 5, 1))
    x_test = np.reshape(x_test, (len(x_test), 200, 5, 1))
    return x_test, x_train


if __name__ == '__main__':
    test, train = get_temperature()
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(train.tolist(), f, ensure_ascii=False, indent=4)
