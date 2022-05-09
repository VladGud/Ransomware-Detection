import matplotlib.pyplot as plt
import threading
import time
import sys
sys.path.append('etw')

import etw
from etw.evntrace import TRACE_LEVEL_INFORMATION

from Serializer import *
from parallel import EventHandler, MyConcurrentCollection, EventDictionary

class SystemDetect:
    def __init__(self):
        self.session_length = 5
        self.col = MyConcurrentCollection()

        # Создание фильтра событий по Task Name
        ed = EventDictionary()
        task_filters = list(ed.maskProcess.keys()) + list(ed.maskFile.keys())[:len(ed.maskFile) - 2]
        del task_filters[1::2]
        threads_count = 1

        # Обработчики событий
        self.handlers = [EventHandler(self.col) for _ in range(threads_count)]

        # Провайдеры подключаемые к сессии
        providers = [etw.ProviderInfo('Microsoft-Windows-Kernel-File',
                                      etw.GUID("{EDD08927-9CC4-4E65-B970-C2560FB5C289}"),
                                      level=TRACE_LEVEL_INFORMATION),
                     etw.ProviderInfo('Microsoft-Windows-Kernel-Process',
                                      etw.GUID("{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}"),
                                      level=TRACE_LEVEL_INFORMATION)]
        # Создание сессии
        self.job = etw.ETW(session_name="MyRansomwareDetectSession", providers=providers, event_callback=self.col.append,
                      task_name_filters=task_filters)

    def star_etw(self):
        # Запуск сессии ETW
        print("start")
        self.job.start()
        time.sleep(self.session_length)
        self.job.stop()
        print("stop")

    def start_handlers(self):
        # Запуск обработчиков событий
        for handler in self.handlers:
            handler.start()

        for handler in self.handlers:
            handler.join()


    def print_collection(self):
        # Вывод собранных данных в консоль
        if not self.col.empty():
            print(self.col.print_collection())
        print(self.col)

    def serialize_collection(self):
        # Сохранение собранных данных в файл
        ser = Serializer()
        ser.serialize("EventLog", list(self.col.collection.queue))

    def print_time_with_operation(self):
        for handler in self.handlers:
            x = []
            keys = handler.ed.dfTimeRead.keys()
            for key in keys:
                plt.title(str(key))
                y = [i for i in handler.ed.dfTimeRead[key]]
                x = [i for i in range(len(y))]
                plt.plot(x, y)
                plt.show()

    def run(self):
        thread_etw = threading.Thread(target=self.star_etw, args=())
        thread_handlers = threading.Thread(target=self.start_handlers, args=())
        thread_etw.start()
        thread_handlers.start()
        thread_etw.join()
        thread_handlers.join()

if __name__ == "__main__":
    systemDetect = SystemDetect()
    systemDetect.run()