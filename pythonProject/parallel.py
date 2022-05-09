import pickle
import queue
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
pd.options.display.max_seq_items = None


class MyConcurrentCollection:
    def __init__(self):
        self.collection = queue.Queue()

    def append(self, x):
        self.collection.put(x)

    def pop(self):
        return self.collection.get()

    def __len__(self):
        return self.collection.qsize()

    def __str__(self):
        return f"{len(self)}"

    def print_collection(self):
        return self.collection.queue

    def empty(self):
        return self.collection.empty()

# Класс для хранения анализируемых данных
class EventDictionary:
    def __init__(self):

        # Словарь подстановщик событий Microsoft-Windows-Kernel-File
        self.maskFile = {
            "OPERATIONEND": 0,
            24: 1,

            "CLOSE": 2,
            14: 3,

            "CREATE": 4,
            12: 5,

            "CLEANUP": 6,
            13: 7,

            "READ": 8,
            15: 9,

            "CREATENEWFILE": 10,
            30: 11,

            "WRITE": 12,
            16: 13,

            "NAMECREATE": 14,
            10: 15,

            "SETINFORMATION": 16,
            17: 17,

            "RENAME": 18,
            19: 19,

            "RENAMEPATH": 20,
            27: 21,

            "NAMEDELETE": 22,
            11: 23,

            "SETDELETE": 24,
            18: 25,

            "DELETEPATH": 26,
            26: 27,

            "SETSECURITY": 28,
            31: 29,

            "READSIZE": 30,

            "WRITESIZE": 31,

        }

        # Словарь подстановщик событий Microsoft-Windows-Kernel-Process
        self.maskProcess = {
            "PROCESSSTART": 32,
            1: 33,

            "PROCESSSTOP": 34,
            2: 35,

            'THREADSTART': 36,
            3: 37,

            'THREADSTOP': 38,
            4: 39,

            'IMAGELOAD': 40,
            5: 41,

            'IMAGEUNLOAD': 42,
            6: 43,
        }

        # Сбор ключевых событий, подсчет количества каждого события и количества прочитанных/записанных байт
        self.dfMainInfo = defaultdict(lambda: np.zeros(len(self.maskProcess) + len(self.maskFile)))

        # Сбор временных интервалов между соседними операциями чтения/записи
        self.dfTimeRead = defaultdict(list)
        self.dfTimeWrite = defaultdict(list)

        # Сохранение времени последней чтения/записи
        self.lastWrite = defaultdict(lambda: -1)
        self.lastRead = defaultdict(lambda: -1)

        # Загрузка службы выявления неизвестных ранее Ransomware, при анализе dfMainInfo
        with open("data\\isolation_forest_model.pkl", 'rb') as file:
            self.model = pickle.load(file)

    def append(self, pid, task, event, provider, io_size, timestamp):
        try:
            # Определение какого поставщика обрабатывается событие и подсчет количества подобных событий
            if provider != '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}':
                self.dfMainInfo[pid][self.maskFile[task]] += 1
                self.dfMainInfo[pid][self.maskFile[event]] += 1

                # Сохранение количества записанных байт и времени
                if task == 'WRITE':
                    self.dfMainInfo[pid][self.maskFile["WRITESIZE"]] += int(io_size, 0)
                    self.dfTimeWrite[pid].append((timestamp - self.lastWrite[pid]) if self.lastWrite[pid] != -1 else 0)
                    self.lastWrite[pid] = timestamp

                # Сохранение количества прочитанных байт и времени
                if task == 'READ':
                    self.dfMainInfo[pid][self.maskFile["READSIZE"]] += int(io_size, 0)
                    self.dfTimeRead[pid].append((timestamp - self.lastRead[pid]) if self.lastRead[pid] != -1 else 0)
                    self.lastRead[pid] = timestamp
            else:
                self.dfMainInfo[pid][self.maskProcess[task]] += 1
                self.dfMainInfo[pid][self.maskProcess[event]] += 1
        except KeyError:
            return

    def predict(self):
        return [[key, self.model.predict([self.dfMainInfo[key]])] for key in self.dfMainInfo]

# Класс для обработки собранных событий ETW
class EventHandler(threading.Thread):
    def __init__(self, collection: MyConcurrentCollection):
        threading.Thread.__init__(self)
        self.daemon = True
        self.collection = collection
        self.ed = EventDictionary()

    def run(self):
        ex = 0
        c = 0
        count = 0

        while True:
            if not self.collection.empty():
                data = self.collection.pop()
                c += 1
                # Выделение ключевых данных из событий
                pid = data[1]['EventHeader']['ProcessId']
                if pid > 0:
                    try:
                        process = psutil.Process(pid)
                        if process.name() != "System":
                            task = data[1]['Task Name']
                            # Сохранение выделенных ключевых данных
                            self.ed.append(process.parents()[0].exe() if len(process.parents()) > 0 else process.exe(),
                                            task,
                                            data[0],
                                            data[1]['EventHeader']['ProviderId'],
                                            data[1]['IOSize'] if task == "READ" or task == "WRITE" else 0,
                                            data[1]['EventHeader']["TimeStamp"])

                    # Процесс остановлен, можно освободить собранные данные по процессу.
                    except psutil.NoSuchProcess:
                        print(f"dropped {pid}")
                        self.ed.dfMainInfo[pid] = np.zeros(len(self.ed.maskProcess) + len(self.ed.maskFile))
                        self.ed.dfTimeRead[pid] = []
                        self.ed.dfTimeWrite[pid] = []
                        self.ed.lastWrite[pid] = -1
                        self.ed.lastRead[pid] = -1

                # Обработано уже достаточное количество для начала анализа
                if c == 1000:
                    count += c
                    c = 0
                    print(self.ed.predict(), count)

            # Ожидание заполнение очереди новыми событиями.
            else:
                time.sleep(0.1)
                ex += 1
                if ex > 100:
                   print(self.ed.predict())
                   return


