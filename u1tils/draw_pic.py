import csv
from matplotlib import pyplot as plt
import mpl_toolkits.axisartist as axisartist

fig = plt.figure(dpi=128, figsize=(8, 10))
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
filename = './data_results/L1_4MR_10.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    highs1 = []
    for row in reader:
        highs1.append(float(row[1]))
filename = './data_results/L1_5MR_10.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    highs2 = []
    for row in reader:
        highs2.append(float(row[1]))
filename = './data_results/L1_6MR_10.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    highs3 = []
    for row in reader:
        highs3.append(float(row[1]))
filename = './data_results/L1_7MR_10.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    highs4 = []
    for row in reader:
        highs4.append(float(row[1]))

l1, = plt.plot(highs1, c='gray')
l2, = plt.plot(highs2, c='blue')
l3, = plt.plot(highs3, c='green')
l4, = plt.plot(highs4, c='red')
plt.legend(handles=[l4, l3, l2, l1], labels=['7 TSRBs', '6 TSRBs', '5 TSRBs', '4 TSRBs'], loc='upper right')
plt.title('Sampling Ratio = 0.10', fontsize=24)
plt.xlabel('Epoch Number in Training', fontsize=30, fontweight='heavy')
plt.ylabel('PSNR(dB)', fontsize=24, fontweight='heavy')
ax.axis['top'].set_visible(False)
ax.axis['right'].set_visible(False)
ax.axis["bottom"].set_axisline_style("-|>", size=1.0)
ax.axis["left"].set_axisline_style("-|>", size=1.0)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.xlim([20,100])
plt.ylim([32.5,34.5])


plt.show()
