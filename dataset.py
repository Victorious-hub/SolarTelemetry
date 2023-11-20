import csv
import json
import os.path
import re
from collections import defaultdict
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
    get_days = set()
    timestamp_sum = defaultdict(float)

    structure = ['timestamp', 'temperature', 'voltage', 'current', 'irradiation']
    x_train = []
    all_dataset = []
    ald = []
    #all_currents = []

    for filename in panels:
        path_to_filename = os.path.join(path, filename)
        if not check_file_exist(path_to_filename):
            path_to_filename = os.path.join(path, filename.replace('.', '_'))
        if os.path.exists(f'{path_to_filename}.csv'):
            with open(f'{path_to_filename}.csv') as f:
                array_irradiation = []
                all_currents = []
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
                    all_currents.append(current)
                    irradiating = float(r['irradiation'])
                    array_irradiation.append(irradiating)

                    pattern = r'^\w{3} \w{3} \d{2} \d{4}'
                    match = re.match(pattern, r['utc_ts'])
                    date_only = match.group()

                    get_days.add(date_only)

                    timestamp_sum[date_only] += current

                    voltage = float(r['voltage'])
                    temperature = float(r['temperature']) + 273.2
                    if prev_day != time.day:
                        counter = 0
                        prev_day = time.day
                        prev_irr = 0
                        prev_curr = 0
                        prev_index = -1

                        if wtf_image and 100 <= len(wtf_image) <= 160:
                            i = len(wtf_image)
                            s = wtf_image[i - 1][0] + 1
                            while i < 160:
                                rx = [s, 123, 123, 123, 123]
                                s = s + 1
                                wtf_image.append(rx)
                                # ald.append(rx)
                                i = i + 1

                        # while len(wtf_image) > 160:
                        #     #if wtf_image and len(wtf_image) > 160:
                        #         gap = len(wtf_image) - 160
                        #         if gap > 160:
                        #             gap = 160
                        #         mid = int(len(wtf_image) / 2)
                        #         wtf_image_tmp = []
                        #         ald_tmp = []
                        #         index = mid - (1 + 2 * (int((gap - 1) / 2)))
                        #         i = 0
                        #         while gap > 0:
                        #             if i != index:
                        #                 wtf_image_tmp.append(wtf_image[i])
                        #                 ald_tmp.append(ald[i])
                        #             else:
                        #                 index = index + 2
                        #                 gap = gap - 1
                        #             i = i + 1
                        #         wtf_image = wtf_image_tmp
                        #         ald = ald_tmp
                        if wtf_image and len(wtf_image) == 160:
                            x_train.append(wtf_image)
                            all_dataset.extend(ald)
                        ald = []
                        wtf_image = []

                    if 8 <= time.hour < 20 and 360 <= irradiating <= 1500 and (len(all_currents) > 1 and all_currents[-2] != 0 and current/all_currents[-2] >= 0.5):  # and temperature >= 25 and temperature <= 400
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
                                # val[3] = val[3]
                                wtf_image.append(val)
                                ald.append(r)
                                prev_irr = irradiating
                                prev_curr = current
                                prev_index = index

   # for day, timestamp_sum_value in timestamp_sum.items():
   #     print(f"Day: {day}, Timestamp Sum: {timestamp_sum_value}")

    return x_train, all_dataset, all_currents, array_irradiation


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
    # all_dataset = []
    for _, dirnames, filenames in walk(path_to_dataset):
        for dir_name in dirnames:
            print(dir_name, dirnames.index(dir_name) + 1, '/', len(dirnames))
            path = os.path.join(path_to_dataset, dir_name, temperature_dir_name)
            panels = get_list_panels()
            x_tr, a_dr, curr, irr = get_panels_dataset(path, panels)
            print(str(len(x_tr)) + ' test records found')
            x_te, _, curr2, irr2 = get_panels_dataset(path, test_panel_names)
            print(str(len(x_te)) + ' test records found')

            x_train.extend(x_tr)
            x_test.extend(x_te)
            # all_dataset.extend(a_dr)
        break

    # Remove arrays with at least one element equal to zero

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = np.reshape(x_train, (len(x_train), 160, 5, 1))
    x_test = np.reshape(x_test, (len(x_test), 160, 5, 1))

    return x_test, x_train


if __name__ == '__main__':
    test, train = get_temperature()

    train_data = train.reshape(len(train), -1)

    # non_zero_rows = np.any(train_data != 0, axis=(1, 2))

    # filtered_train_data = train_data[non_zero_rows]

    # condition = np.all(filtered_train_data == [False, True, True, True, True], axis=2)
    # filtered_train_data = filtered_train_data[~condition]

    with open('data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in train_data:
            writer.writerow(row.flatten())
