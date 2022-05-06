import pickle
import queue
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
from numpy import uint64
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report

pd.options.display.max_seq_items = None
import dask.dataframe as dd


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


class EventDictionary:
    def __init__(self):
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
        self.maskProcess = {
            "PROCESSSTART": 0,
            1: 1,

            "PROCESSSTOP": 2,
            2: 3,

            'THREADSTART': 4,
            3: 5,

            'THREADSTOP': 6,
            4: 7,

            'IMAGELOAD': 8,
            5: 9,

            'IMAGEUNLOAD': 10,
            6: 11,
        }

        self.dfFile = defaultdict(lambda: np.zeros(len(self.maskFile)))
        self.dfProcess = defaultdict(lambda: np.zeros(len(self.maskProcess)))

        self.dfTimeRead = defaultdict(list)
        self.dfTimeWrite = defaultdict(list)

        self.lastWrite = -1
        self.lastRead = -1

        with open("data\\isolation_forest_model.pkl", 'rb') as file:
            self.model = pickle.load(file)

    def append(self, pid, task, event, provider, io_size, timestamp):
        try:
            if provider != '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}':
                self.dfFile[pid][self.maskFile[task]] += 1
                self.dfFile[pid][self.maskFile[event]] += 1

                if task == 'WRITE':
                    self.dfFile[pid][self.maskFile["WRITESIZE"]] += int(io_size, 0)
                    self.dfTimeWrite[pid].append(timestamp - (self.lastWrite if self.lastWrite != -1 else 0))
                    self.lastWrite = timestamp

                if task == 'READ':
                    self.dfFile[pid][self.maskFile["READSIZE"]] += int(io_size, 0)
                    self.dfTimeRead[pid].append(timestamp - (self.lastRead if self.lastRead != -1 else 0))
                    self.lastRead = timestamp

            else:
                self.dfProcess[pid][self.maskProcess[task]] += 1
                self.dfProcess[pid][self.maskProcess[event]] += 1

        except KeyError:
            return



    def predict(self):
        return [[key, self.model.predict([self.dfFile[key]])] for key in self.dfFile]


class Consumer(threading.Thread):
    def __init__(self, collection: MyConcurrentCollection):
        threading.Thread.__init__(self)
        self.daemon = True
        self.collection = collection
        self.times = []
        self.ed = EventDictionary()
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

    def run(self):
        self.times.clear()
        ex = 0
        c = 0
        count = 0

        while True:
            if not self.collection.empty():
                data = self.collection.pop()
                c += 1
                pid = data[1]['EventHeader']['ProcessId']
                if pid > 0:
                    self.times.append(time.time())
                    try:
                        process = psutil.Process(pid)
                        if process.name() != "System":
                            task = data[1]['Task Name']
                            self.ed.append(process.parents()[0].exe() if len(process.parents()) > 0 else process.exe(),
                                           task,
                                           data[0],
                                           data[1]['EventHeader']['ProviderId'],
                                           data[1]['IOSize'] if task == "READ" or task == "WRITE" else 0,
                                           data[1]['EventHeader']["TimeStamp"])

                    except psutil.NoSuchProcess:
                        print(f"dropped {pid}")
                        self.ed.dfFile[pid] = np.zeros(44)
                        self.ed.dfProcess = np.zeros(12)
                        self.ed.dfTimeRead = defaultdict(list)
                        self.ed.dfTimeWrite = defaultdict(list)
                        self.ed.lastWrite = -1
                        self.ed.lastRead = -1

                    self.times.append(time.time())

                if c == 1000:
                    count += c
                    c = 0
                    #print(ed.predict(), count)

            else:
                time.sleep(0.1)
                ex += 1
                if ex > 100:
                   #print(ed.predict())
                    return


