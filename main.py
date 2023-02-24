import time
import numpy as np
from threading import Thread
from log import Log
from memory_getter import MemoryGetter
from client_mem_vm import ClientMemVM
from apache_benchmark import Benchmark
from inferance import Inferance
from colorama import init, Fore, Back, Style

DURATION = 99999
FINESSE = 0.5
TR = 0
SUEIL_TR = 500


b = Benchmark()
log_ = Log()

def run_bench():
    global time_reponse
    time_reponse=0
    while True:
        time_reponse = b.start_benchmark()


def change_limit_cgroup_file(cgroup_limit):
    with open(r"/sys/fs/cgroup/machine.slice/machine-qemu\x2d5\x2dubuntu20.04.scope/libvirt/memory.max","w") as fmax:
        fmax.write(cgroup_limit)
    

class Mechanism():
    bash_tmp = []
    bash = []
    curr_tr_value = 0

    def __init__(self,memorygetter,clientMemVm):
        self.memorygetter = memorygetter
        self.clientMemVm = clientMemVm
        self.infer = Inferance()

    def run(self):

        start_time = time.process_time()

        t = round(time.process_time()-start_time,2) 
        Thread(target=run_bench).start()
        while t<DURATION: 
            diff_t = round(time.process_time() - t - start_time,2)
            if diff_t>=FINESSE and time_reponse > 0 :#we wait for the first value of the benchmark to perform an inference
                t+= diff_t
                self.update_bash()
                self.do_inferance()
                print(Style.BRIGHT  + Fore.GREEN + "[t = {} s]".format(int(t)),end="\r")

            

    
    def update_bash(self):
        if np.array(self.bash_tmp).shape[0]<25:
            self.bash_tmp.append([
                self.memorygetter.get_mem_proc() / 1048576 ,
                self.clientMemVm.get_value() / 1024,
                self.memorygetter.get_limit_cgroup() / 1048576,
                time_reponse
            ])
        else :
            self.bash = self.bash_tmp
            self.bash_tmp = []
          
    def do_inferance(self):
        # we do prediction only when we have a new time reponse value
        # that prevent us to cut lower the cgroup limit too frequently and make a prediction with tr value that are not descriptive of the VM state
        if self.curr_tr_value != time_reponse and np.array(self.bash).shape[0]==25: 
            self.do_predict()
            self.curr_tr_value = time_reponse


    def do_predict(self):
        x_bash = np.array([self.bash])
        tr_predict = self.infer.predict(x_bash)
        log_.output("(tr predict is - "+str(tr_predict[0])+") (tr current is - "+str(self.curr_tr_value)+")")
        self.change_limit_cgroup(tr_predict[0])


    def change_limit_cgroup(self,tr_predict):
        if tr_predict < SUEIL_TR:
            new_limit = int(self.memorygetter.get_limit_cgroup() * 0.75)
            change_limit_cgroup_file(str(new_limit))
            log_.debug("limit_cgroup - "+str(int(self.memorygetter.get_limit_cgroup()/1024))+" kB")
            print(Style.BRIGHT  + Fore.WHITE + "CUT MEMORY")
        else :
            new_limit = int(self.memorygetter.get_limit_cgroup() * 1.25)
            change_limit_cgroup_file(str(new_limit))
            log_.debug("limit_cgroup - "+str(int(self.memorygetter.get_limit_cgroup()/1024))+" kB")
            print(Style.BRIGHT  + Fore.WHITE + "ADD MEMORY")



if __name__=="__main__":
    
    memorygetter = MemoryGetter()
    clientMemVm = ClientMemVM()
    clientMemVm.connect()
    time.sleep(2)


    mechanism = Mechanism(memorygetter,clientMemVm)
    mechanism.run()
    










