import os
import matplotlib.pyplot as plt

directory = 'third/'
k_2 = []
avg_2 = []
tupl_plot = []

for filename in os.listdir(directory):
    third_value = 0
    curr_k = 0
    with open(directory+filename, "r") as file:
        temp_filename = os.path.splitext(filename)
        t = temp_filename[0].split("_")
        curr_k = t[1]
        if t[2] != "2":
            continue
        for line in file.readlines():
            third_value += int(line.split(" ")[2])

    average = third_value / 10
    tupl_plot.append((int(curr_k),average))

ordered_plot = sorted(tupl_plot, key=lambda x: x[0], reverse=False)
print(ordered_plot)
for k,avg in ordered_plot:
    k_2.append(k)
    avg_2.append(avg)


plt.plot(k_2, avg_2, marker='o', linestyle='-', color='b', label='Third')
# Titoli degli assi e del grafico
plt.xlabel('size_seed_set')
plt.ylabel('avg_influenced')
plt.legend()
plt.grid(True)

plt.title('Threshold = ' + str(2))

# Visualizzazione del grafico
plt.show()


