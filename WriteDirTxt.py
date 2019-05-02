# -*- coding:utf-8 -*-
import  pickle
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

root_dir=r"/Users/ageliss/Downloads/Ocean_Predict_Center_data"
with open("./OutputAsTxt.txt", "w") as output:
    output.write("台风名称"+"\t"+"台风编号"+"\t"+"预测机构"+"\t"+"时间"+"\t"+"实况台风纬度"+"\t"+"实况台风经度"+"\t"+"台风最低气压"+"\t"
                 +"台风最大风速"+"\t"+"24lat"+"\t"+"24lon"+"\t"+"24pre"+"\t"+"24wind_seed"+"\t"+"48lat"+"\t"+"48lon"+"\t"+"48pre"+"\t"+
                 "48wind_seed"+"\t"+"72lat"+"\t"+"72lon"+"\t"+"72pre"+"\t"+"72wind_seed"+"\n")
    list_dirs = os.walk(root_dir)
    res = pd.DataFrame(columns=['ty_name', 'ty_ID', 'agency', 'time', 'lon', 'lat', 'MSLP', 'MaxWind', '24lat', '24lon',
                                '24MSLP', '24MaxWind', '48lat', '48lon', '48MSLP', '48MaxWind',
                                '72lat', '72lon', '72MSLP', '72MaxWind'])
    res1 = res
    for root, dirs, files in list_dirs:
        for d in dirs:
            dir=os.path.join(root, d)
            for info in os.listdir(dir):
                if(info.endswith(".wrf") or info.endswith(".coawst") or info.endswith('.BAB')):
                    '''Write a seperate DataFrame for this kinds'''
                    continue
                file_name = os.path.join(dir, info)  # 将路径与文件名结合起来就是每个文件的完整路径
                with open(file_name, 'r') as file:  # 读取文件内容
                    print('Reading: ', file_name)
                    line = file.readline()
                    # print(line.__str__())
                    eachline = line.split()

                    if eachline == []:
                        continue
                    forecasting_agency = eachline[0]
                    ty_name = eachline[1]
                    ty_ID = eachline[2]
                    # print(line)
                    for lines in file:
                        # print(lines)
                        if lines == "":
                            break
                        else:
                            eachline = lines.split()

                            output.write(ty_name + "\t" + ty_ID + "\t" + forecasting_agency + "\t" + eachline[0] + "-" +
                                         eachline[1] + "-" + eachline[2] + "时" + "\t")
                            for i in range(3,eachline.__len__()):
                                output.write(eachline[i]+"\t")
                            time = datetime(int('20'+ty_ID[0:2]), int(eachline[0]), int(eachline[1]), int(eachline[2]))
                            data = [ty_name, ty_ID, forecasting_agency, time]
                            for j in range(3, 19):
                                data.append(eachline[j])
                            df1 = pd.DataFrame([data],
                                               columns=['ty_name', 'ty_ID', 'agency', 'time', 'lon', 'lat', 'MSLP',
                                                        'MaxWind',
                                                        '24lat', '24lon', '24MSLP', '24MaxWind', '48lat', '48lon',
                                                        '48MSLP', '48MaxWind', '72lat', '72lon', '72MSLP', '72MaxWind'])
                            res = res.append(df1, ignore_index=True)

                            output.write("\n")
    print(res)
                    # sys.exit()

