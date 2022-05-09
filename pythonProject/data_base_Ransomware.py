import numpy as np
import pandas as pd
from Serializer import Serializer
from parallel import EventDictionary
from collections import defaultdict

# Id процессов Ransomware
numbers = [
"0 - 19324",
"1 - 16488",
"2 - 6028",
"3 - 1244"
]

ser = Serializer()
ed = EventDictionary()

# Создание заголовков столбцов
columns_task = ["pid", "label"] + list(ed.maskFile.keys())[:len(ed.maskFile) - 2]
#del columns_task[3::2]
columns_task += list(ed.maskFile.keys())[len(ed.maskFile) - 2: len(ed.maskFile)]
columns_procces = list(ed.maskProcess.keys())
#del columns_procces[1::2]
columns_task += columns_procces

# Доп столбец для обучения
label = defaultdict(lambda: 0)

# Обработка событий из json файлов
for i in numbers:
    num, pidx = i.split(" - ")
    json = ser.deserialize(f"data\\EventLog{num}.json")
    print(i, len(json))
    for data in json:
        _pid = str(data[1]['EventHeader']['ProcessId'])
        pid = _pid + f" - {num}"
        new_data = {'EventId': data[0], 'Task': data[1]['Task Name'],
                    'Base Process Exe': pid, 'ProviderId': data[1]['EventHeader']['ProviderId'],
                    'IOSize': data[1]['IOSize'] if data[1]['Task Name'] == "READ" or data[1]['Task Name'] == "WRITE" else 0,
                    'TimeStamp': data[1]['EventHeader']["TimeStamp"]
        }
        ed.append(new_data['Base Process Exe'], new_data['Task'], new_data['EventId'], new_data['ProviderId'], new_data['IOSize'], new_data['TimeStamp'])
        if pidx == _pid:
            label[pid] = 1
        else:
            label[pid] = 0

# Сохранение в csv файл обработанных данных из EventDictionary
key = ed.dfMainInfo.keys()
arr = []
for k in key:
    df_list = list(ed.dfMainInfo[k])
    #del df_list[len(ed.maskFile)::2]
    #del df_list[1:len(ed.maskFile) - 2:2]
    arr.append([k, label[k]] + list(df_list))
pd.DataFrame(arr, columns=columns_task).to_csv("main11.csv")
