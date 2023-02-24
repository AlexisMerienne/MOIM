
import re
from subprocess import PIPE, run


MOTIF_RE_FLOAT = "\d+"


class Benchmark():

    def __init__(self):
        pass

    def start_benchmark(self):
        res_time = 0
        benchmark_cmd = "ab -n 100000 -c 500 http://192.168.122.49:80/"
        result = run(benchmark_cmd,stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
        for l in result.stdout.splitlines():
            if "(longest request)" in l:
                res_time = int(re.findall(MOTIF_RE_FLOAT,l)[1])
                break
            
        return res_time
