import sys
sys.path.append('etw')
from Serializer import *
import etw
from etw.evntrace import TRACE_LEVEL_INFORMATION
import time
from parallel import Consumer, MyConcurrentCollection, EventDictionary
import matplotlib.pyplot as plt


session_length = 1
col = MyConcurrentCollection()


def init_session_and_consumer():
    ed = EventDictionary()
    task_filters = list(ed.maskProcess.keys()) + list(ed.maskFile.keys())[:len(ed.maskFile) - 2]
    del task_filters[1::2]
    threads_count = 1
    consumers = [Consumer(col) for _ in range(threads_count)]
    providers = [etw.ProviderInfo('Microsoft-Windows-Kernel-File',
                                  etw.GUID("{EDD08927-9CC4-4E65-B970-C2560FB5C289}"),
                                  level=TRACE_LEVEL_INFORMATION),
                 etw.ProviderInfo('Microsoft-Windows-Kernel-Process',
                                  etw.GUID("{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}"),
                                  level=TRACE_LEVEL_INFORMATION)]
    job = etw.ETW(session_name="MyRansomwareDetectSession", providers=providers, event_callback=col.append,
                  task_name_filters=task_filters)

    def start_session_and_consumer():
        print("start")
        job.start()
        time.sleep(session_length)
        job.stop()
        print("stop")

        if not col.empty():
            print(col.print_collection())
        print(col)

        sx = 0
        ser = Serializer()
        ser.serialize("EventLog", list(col.collection.queue))

        for consumer in consumers:
            consumer.start()

        for consumer in consumers:
            consumer.join()

        for consumer in consumers:
            sx += 1
            x = []
            for i in range(0, len(consumer.times), 2):
                x.append(consumer.times[i + 1] - consumer.times[i])

            plt.plot([i for i in range(len(x))], x)
            plt.show()

            #sx = max(sx, sum(x))

        print(sx)

    start_session_and_consumer()


init_session_and_consumer()

# ser = Serializer()
# num = 16
# ser.serialize(f"data\\test{num}.json", list(col.collection.queue))

# 0 - 6848
# 1 - 13616
# 2 - 24952
# 3 - 23140
# 4 - 7440
# 5 - 15276
# 6 - 26436
# 7 - 23352
# 8 - 23452
# 9 - 13604
# 10 - 11704
# 11 - 25496
# 12 - 13272
# 13 - 3916
# 14 - 6116
# 15 - 13972 !!!

"""
logman query providers
logman start pstrace -p Microsoft-Windows-Kernel-File -o log.etl -ets
logman update pstrace -p Microsoft-Windows-Kernel-File 0x0000000000000200 -ets
logman stop pstrace -ets
tracerpt log.etl -o log.xml
logman query providers Microsoft-Windows-Kernel-File
wevtutil gp Microsoft-Windows-Kernel-File /ge /gm
"""

"""
(15, {'EventHeader': {'Size': 128, 'HeaderType': 0, 'Flags': 576, 'EventProperty': 0, 
'ThreadId': 11364, 'ProcessId': 11092, 'TimeStamp': 132635050005504896, 

'ProviderId': 
'{EDD08927-9CC4-4E65-B970-C2560FB5C289}',

'EventDescriptor':
{'Id': 15, 'Version': 1, 'Channel': 16, 'Level': 4, 'Opcode': 0, 'Task': 15, 'Keyword': 9223372036854776096}, 
'KernelTime': 41, 'UserTime': 104, 'ActivityId': '{00000000-0000-0000-0000-000000000000}'}, 

'Task Name': 'READ', 'ByteOffset': '0x335', 'Irp': '0xFFFF818599569888', 'FileObject': '0xFFFF81859EFC53F0', 
'FileKey': '0xFFFFC401DFC29830', 'IssuingThreadId': '11364', 'IOSize': '0x200', 'IOFlags': '0x0', 
'ExtraFlags': '0x0', 'Description': ''})
"""
