import multiprocessing
import platform
import psutil
from tensorflow.python.client import device_lib

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

p = platform.processor()
mem = psutil.virtual_memory()

print(f'Your CPU is: {p}')
print(f'Your CPU has {multiprocessing.cpu_count()} cores.')
print(f'Total memory: {sizeof_fmt(mem.total)}')
print(mem)

local_device_protos = device_lib.list_local_devices()
gpuList = [x for x in local_device_protos if x.device_type == 'GPU']

print("Installed GPUs:")
for x in gpuList:
  print("{} - {}".format(x.name,x.physical_device_desc))