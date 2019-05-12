# -*- coding:utf-8 -*-
import  pickle
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

root_dir = r"/Users/ageliss/Downloads/Ocean_Predict_Center_data"
list_dirs = os.walk(root_dir)
res = pd.DataFrame(columns=['ty_name', 'ty_ID', 'agency', 'time', 'lon', 'lat', 'MSLP', 'MaxWind', '24lat', '24lon',
                            '24MSLP', '24MaxWind', '48lat', '48lon', '48MSLP', '48MaxWind',
                            '72lat', '72lon', '72MSLP', '72MaxWind'])
for root, dirs, files in list_dirs:
    for d in dirs:
        dir = os.path.join(root, d)
        for info in os.listdir(dir):
            if(info.endswith(".wrf") or info.endswith(".coawst") or info.endswith('.BAB')):
                '''Write a seperate DataFrame for this kinds'''
                continue
            file_name = os.path.join(dir, info)  # 将路径与文件名结合起来就是每个文件的完整路径
            with open(file_name, 'r') as file:  # 读取文件内容
                print('Reading: ', file_name)
                line = file.readline()
                eachline = line.split()

                if eachline == []:
                    continue
                forecasting_agency = eachline[0]
                ty_name = eachline[1]
                ty_ID = eachline[2]
                if info.endswith('.PGT'):
                    file_name2 = os.path.join(dir, info[:-4]+'.BAB')
                    # print(file_name2)
                    try:
                        with open(file_name2, 'r') as file2:
                            line2 = file2.readline().split()
                            ty_ID = line2[2]
                            print('Use BAB ty_ID instead')
                    except:
                        continue
                year = eachline[4]
                # print(line)
                for lines in file:
                    # print(lines)
                    if lines == "":
                        break
                    else:
                        eachline = lines.split()

                        try:
                            time = datetime(int(year), int(eachline[0]), int(eachline[1]), int(eachline[2]))
                        except:
                            print('Wrong reading: ', file_name)
                            continue
                        data = [ty_name, ty_ID, forecasting_agency, time]
                        for j in range(3, 19):
                            data.append(eachline[j])
                        df1 = pd.DataFrame([data],
                                           columns=['ty_name', 'ty_ID', 'agency', 'time', 'lon', 'lat', 'MSLP',
                                                    'MaxWind',
                                                    '24lat', '24lon', '24MSLP', '24MaxWind', '48lat', '48lon',
                                                    '48MSLP', '48MaxWind', '72lat', '72lon', '72MSLP', '72MaxWind'])
                        res = res.append(df1, ignore_index=True)
                        # output.write("\n")
res.to_csv('Typhoon_data3.csv')
print(res)

