import re

MOTIF = "\d+"
VM_CGROUP_DIR = "/sys/fs/cgroup/machine.slice/machine-qemu\\x2d5\\x2dubuntu20.04.scope/memory.current"
VM_CGROUP_MAX_DIR = "/sys/fs/cgroup/machine.slice/machine-qemu\\x2d5\\x2dubuntu20.04.scope/libvirt/memory.max"
MEM_VM_DIR = "~/Documents/surbooking-experiments/sockets"

class MemoryGetter():

    def __init__(self):
        None

    def get_mem_used(self):
        memtot = memfree = buff = cached = 0
        with open("/proc/meminfo","r") as f_meminfo:
            lines = f_meminfo.readlines()
            for  l in lines:
                if "MemTotal" in l:
                    memtot = int(re.findall(MOTIF,l)[0])
                if "MemFree" in l:
                    memfree = int(re.findall(MOTIF,l)[0])
                if "Buffers" in l:
                    buff = int(re.findall(MOTIF,l)[0])
                if "Cached" in l:
                    cached = int(re.findall(MOTIF,l)[0])

        memused = memtot - memfree - buff - cached

        return memused

    def get_swap_used(self):
        swaptot = swapfree = 0
        with open("/proc/meminfo","r") as f_meminfo:
            lines = f_meminfo.readlines()
            for  l in lines:
                if "SwapTotal" in l:
                    swaptot = int(re.findall(MOTIF,l)[0])
                if "SwapFree" in l:
                    swapfree = int(re.findall(MOTIF,l)[0])
   
        swap_used = swaptot - swapfree
        return swap_used

    def get_mem_proc(self):
        mem_proc = 0
        with open(VM_CGROUP_DIR,'r') as f_memcurrent:
            mem_proc = int(f_memcurrent.read())
        return mem_proc

    def get_limit_cgroup(self):
        limit_cgroup = 0
        with open(VM_CGROUP_MAX_DIR,'r') as f_mem_max:
            limit_cgroup = int(f_mem_max.read())
        return limit_cgroup


        
 


