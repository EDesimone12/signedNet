import subprocess
import argparse

algorithms = ['tss', 'csm']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', dest='a', action='store',
                        default='', type=str, help='name of the algorithm')
    args = parser.parse_args()

    script_path = "graph.py"

    num_executions = 10

    for algorithm in algorithms:
        for k in [70, 90, 110]:
            for t in [2, 3]:
                command = f"python3.7 graph.py -k {k} -t {t} -a {algorithm}"

                for i in range(num_executions):
                    subprocess.call(command, shell=True)