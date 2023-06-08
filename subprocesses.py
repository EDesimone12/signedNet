import subprocess
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', dest='a', action='store',
                        default='', type=str, help='name of the algorithm')
    args = parser.parse_args()

    script_path = "graph.py"

    num_executions = 10

    for k in [5, 10, 20, 30, 40, 50]:
        for t in [2, 3]:
            command = f"python graph.py -k {k} -t {t} -a {args.a}"

            for i in range(num_executions):
                subprocess.call(command, shell=True)
