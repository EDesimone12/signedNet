import os
import matplotlib.pyplot as plt
import argparse


def custom_avg(algorithm, threshold):
    directory = f'{algorithm}/'
    k_2 = []
    avg_2 = []
    tupl_plot = []

    for filename in os.listdir(directory):
        third_value = 0
        curr_k = 0
        with open(directory + filename, "r") as file:
            temp_filename = os.path.splitext(filename)
            t = temp_filename[0].split("_")
            curr_k = t[1]
            if t[2] != threshold:
                continue
            for line in file.readlines():
                third_value += (int(line.split(" ")[2]) + int(curr_k))

        average = third_value / 10
        tupl_plot.append((int(curr_k), average))

    ordered_plot = sorted(tupl_plot, key=lambda x: x[0], reverse=False)
    print(ordered_plot)
    for k, avg in ordered_plot:
        k_2.append(k)
        avg_2.append(avg)

    with open(f"avg_{algorithm}_{threshold}.txt", "a") as file:
        for k, avg in ordered_plot:
            file.write(f"{k} {avg} \n")


###########

def custom_plot(threshold):
    file_list = None

    if threshold == 2:
        file_list = ["avg_csm_2", "avg_tss_2", "avg_third_2"]
    else:
        file_list = ["avg_csm_3", "avg_tss_3", "avg_third_3"]

    colors = ""

    for curr_file in file_list:
        k_vals = []
        avg_vals = []
        name = ""
        with open(curr_file + ".txt", "r") as file:
            temp_filename = os.path.splitext(curr_file)
            t = temp_filename[0].split("_")
            name = t[1]
            for line in file.readlines():
                k_vals.append(int(line.split(" ")[0]))
                avg_vals.append(float(line.split(" ")[1]))
        if name.upper() == "THIRD":
            colors = "g"
        elif name.upper() == "TSS":
            colors = "b"
        elif name.upper() == "CSM":
            colors = "r"

        print("k:", k_vals, "avg", avg_vals)
        plt.plot(k_vals, avg_vals, marker='o', linestyle='-', color=colors, label=name.upper())

    # Titoli degli assi e del grafico
    plt.xlabel('size_seed_set')
    plt.xticks(range(0, 55, 5))
    # plt.yticks(range(3600, 3901, 50))
    plt.ylabel('avg_influenced')
    plt.legend()
    plt.grid(True)

    plt.title('Threshold = ' + str(3))

    # Visualizzazione del grafico
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-type', dest='type', action='store',
                        default='', type=str, help='plot or avg')
    parser.add_argument('-a', dest='type', action='store',
                        default='', type=str, help='algorithm')
    parser.add_argument('-t', dest='type', action='store',
                        default='', type=str, help='threshold')
    args = parser.parse_args()

    if args.type == "plot":
        custom_plot(args.t)
    elif args.type == "avg":
        custom_avg(args.a, args.t)