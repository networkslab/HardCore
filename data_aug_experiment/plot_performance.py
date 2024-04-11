from matplotlib import pyplot as plt
import csv
import os
import numpy as np
from scipy.stats import wilcoxon
plt.rcParams['font.size'] = 15
csv_ = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_bench_rankings_csv.csv'
combined_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined'
original_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_og'
hardsatgen_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin'
# hardsatgen_path =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin'
w2sat_path =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin'

# csv_ = '/home/joseph-c/sat_gen/CoreDetection/training_exp/internal_bench_rankings_full_csv.csv'
# combined_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined'
# original_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og'
# hardsatgen_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc'
# w2sat_path =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc'

csv_ = open(csv_)

reader = csv.reader(csv_)
skip_header=True
combined = []
original = []
x = []
combined_trials=  []
original_trials = []
hardsatgen_trials = []
hsStrict_trials = []
w2_trials = []
for row in reader:
    try: int(row[0])
    except: continue
    
    combined.append(float(row[1]))
    original.append(float(row[2]))
    x.append(int(row[0]))

test = []
test_reader = csv.reader(open('/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/bench_ranking_100.csv'))
for row in test_reader:
    test.append(float(row[0]))

fig1, ax1 = plt.subplots()
ax1.boxplot(test, sym='')


# # x = [x[11]]
# x = x[:2]
# y = [20, 40]
# x = [20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1200]
# x = x[:6]
# y = [100, 200, 300, 400, 500, 600, 700]
# x = [100, 200, 300, 500,  1000]
x = [10, 20, 30, 40]
# x = [1320]
# y = [100, 200, 600]
# x = [20, 50, 100]
# x = [20 40 60 ]
# x = x[:2]
# y = y[:1]
# x = x[:7]
# y = y[:7]
# x= y
# x = x[:1]
# y = y[:3]
# x = [100,200,300]
# y = [100, 50, 20]
# x = x[:2]
# y = y[:2]
# x = [200, 400, 600, 800, 1000]
# x = [100, 200, 300, 400, 500]
for i in x:
    combined_trials.append([])
    # com_csv = open(combined_path + '/bench_ranking_' + str(i) + '_3_layer_600_hidden.csv')
    com_csv = open(combined_path + '/bench_ranking_v9_' + str(i) + '.csv')
    # com_csv = open(combined_path + '/bench_ranking_' + str(i) + '.csv')

    com_reader = csv.reader(com_csv)
    for row in com_reader:
        combined_trials[-1].append(float(row[0]))
for i in x:
    original_trials.append([])
    # og_csv = open(original_path + '/bench_ranking_' + str(i+1) + '_3_layer_600_hidden.csv')
    og_csv = open(original_path + '/bench_ranking_v9_' + str(i) + '.csv')
    # og_csv = open(original_path + '/bench_ranking_' + str(i) + '.csv')
    og_reader = csv.reader(og_csv)
    for row in og_reader:
        original_trials[-1].append(float(row[0]))
for i in x:
    hardsatgen_trials.append([])
    if i == 1445:
        i += 1
    # hs_csv = open(hardsatgen_path + '/bench_ranking_loose_' + str(i) + '.csv')
    hs_csv = open(hardsatgen_path + '/bench_ranking_loose_v9_' + str(i) + '.csv')
    # og_csv = open(original_path + '/bench_ranking_' + str(i) + '.csv')
    hs_reader = csv.reader(hs_csv)
    for row in hs_reader:
        hardsatgen_trials[-1].append(float(row[0]))

for i in x:
    hsStrict_trials.append([])
    if i == 1445:
        i += 1
    # hss_csv = open(hardsatgen_path + '/bench_ranking_strictAgain3_' + str(i) + '.csv')
    hss_csv = open(hardsatgen_path + '/bench_ranking_strict_v9_' + str(i) + '.csv')
    # og_csv = open(original_path + '/bench_ranking_' + str(i) + '.csv')
    hss_reader = csv.reader(hss_csv)
    for row in hss_reader:
        hsStrict_trials[-1].append(float(row[0]))

for i in x:
    w2_trials.append([])
    if i == 1445:
        i += 1
    w2_csv = open(w2sat_path + '/bench_ranking_v9_' + str(i) + '.csv')
    # w2_csv = open(w2sat_path + '/bench_ranking_' + str(i) + '.csv')
    w2_reader = csv.reader(w2_csv)
    for row in w2_reader:
        w2_trials[-1].append(float(row[0]))
fig1, ax1 = plt.subplots()
w_og= []
w_hs = []
w_hss = []
w_w2 = []
for i in range(len(x)):
    # print(np.asarray(original_trials[i]).squeeze().ndim)
    wilcox_og = wilcoxon(x=np.asarray(original_trials[i]).squeeze() ,y=np.asarray(combined_trials[i]).squeeze(), alternative='greater')
    w_og.append([wilcox_og.statistic, np.round(wilcox_og.pvalue, 3)])

    wilcox_hs = wilcoxon(x=np.asarray(hardsatgen_trials[i]).squeeze() ,y=np.asarray(combined_trials[i]).squeeze(), alternative='greater')
    w_hs.append([wilcox_hs.statistic, np.round(wilcox_hs.pvalue, 3)])

    wilcox_hss = wilcoxon(x=np.asarray(hsStrict_trials[i]).squeeze() ,y=np.asarray(combined_trials[i]).squeeze(), alternative='greater')
    w_hss.append([wilcox_hss.statistic, np.round(wilcox_hss.pvalue, 3)])

    wilcox_w2 = wilcoxon(x=np.asarray(w2_trials[i]).squeeze() ,y=np.asarray(combined_trials[i]).squeeze(), alternative='greater')
    w_w2.append([wilcox_w2.statistic, np.round(wilcox_w2.pvalue, 3)])

# print(w)
og_diff = []
for i in range(len(combined_trials)):
    og_diff.append([])
    for j in range(len(combined_trials[i])):        
        og_diff[-1].append(original_trials[i][j] - combined_trials[i][j])

hs_diff = []
for i in range(len(combined_trials)):
    hs_diff.append([])
    for j in range(len(combined_trials[i])):        
        hs_diff[-1].append(hardsatgen_trials[i][j] - combined_trials[i][j])

hss_diff = []
for i in range(len(combined_trials)):
    hss_diff.append([])
    for j in range(len(combined_trials[i])):        
        hss_diff[-1].append(hsStrict_trials[i][j] - combined_trials[i][j])

w2_diff = []
for i in range(len(combined_trials)):
    w2_diff.append([])
    for j in range(len(combined_trials[i])):        
        w2_diff[-1].append(w2_trials[i][j] - combined_trials[i][j])


sidebyside_diff = []
for i in range(len(combined_trials)):
    sidebyside_diff.append(og_diff[i])
    # sidebyside_diff.append(combined_trials[i])
    # sidebyside.append(hardsatgen_trials[i])
    sidebyside_diff.append(hs_diff[i])
    sidebyside_diff.append(hss_diff[i])
    sidebyside_diff.append(w2_diff[i])
xlabels = []
xticks = []
tick = 0
w_og_x = []
w_hs_x = []
w_hss_x = []
w_w2_x = []
for i in (x):
    # xlabels.append(str(i) + ': ' + 'O')
    # xlabels.append(str(i) + ':' + ' G')
    xlabels.append(str(i))
    xlabels.append(str(''))
    xlabels.append(str(''))
    xlabels.append(str(''))
    # xlabels.append(str(''))
    # xticks.append(tick)
    # w_og_x.append(tick)
    # tick += 2
    
    xticks.append(tick)
    
    tick += 2
    w_hs_x.append(tick)
    xticks.append(tick)

    
    tick += 2
    w_hss_x.append(tick)
    xticks.append(tick)

    tick += 2
    w_w2_x.append(tick)
    xticks.append(tick)


    tick += 4
fig1, ax1 = plt.subplots()
ax1.set_ylabel('Method MAE - Our MAE')
ax1.set_xlabel('Size of Original Data Used')
ax1.set_title("Difference in Prediction MAE on Internal LEC Data")

# ax1.set_title("Original (Green) and Augmented (Blue) Predicted Runtime MAE on Tseitin Data")
ax1.set_xticklabels(xlabels, rotation=75)

# ax1.boxplot(sidebyside, sym='', patch_artist=True, positions=xlabels)
# xticks = xticks[:-2]

bplot = ax1.boxplot(sidebyside_diff, sym='', patch_artist=True, positions=xticks)

green = 0
color = ''
for patch in bplot['boxes']:
    if green % 4 == 0:
        color = 'green'
    # if green % 4 == 1:
    #     color = 'blue'
    if green % 4 == 1:
        color = 'red'
    if green % 4 == 2:
        color = 'purple'
    if green % 4 == 3:
        color = 'brown'
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    green += 1


# ax1.scatter(x, combined, label='Original + Synthetic')
# ax1.scatter(x, original, label = 'Original')
# ax1.legend()
# ax1.set_ylabel('Runtime MAE')
# ax1.set_xlabel('Size of Original Data Used')
# ax1.set_title('Runtime Prediction Trained on Original and Augmented Internal LEC Data')
# plt.show()

# fig1, ax1 = plt.subplots()
# ax1.boxplot(combined_trials, sym='')

sidebyside = []
# for i in range(len(original_trials)):
#     sidebyside.append([val for pair in zip(original_trials[i], combined_trials[i]) for val in pair])
for i in range(len(combined_trials)):
    sidebyside.append(original_trials[i])
    sidebyside.append(combined_trials[i])
    sidebyside.append(hardsatgen_trials[i])
    sidebyside.append(hsStrict_trials[i])
    sidebyside.append(w2_trials[i])
# sidebyside = np.stack(sidebyside)
xlabels = []
xticks = []
tick = 0
w_og_x = []
w_hs_x = []
w_hss_x = []
w_w2_x = []
for i in x:
    # xlabels.append(str(i) + ': ' + 'O')
    # xlabels.append(str(i) + ':' + ' G')
    xlabels.append(i)
    xlabels.append(str(''))
    xlabels.append(str(''))
    xlabels.append(str(''))
    xlabels.append(str(''))
    xticks.append(tick)
    w_og_x.append(tick)
    tick += 2
    
    xticks.append(tick)
    
    tick += 2
    w_hs_x.append(tick)
    xticks.append(tick)

    
    tick += 2
    w_hss_x.append(tick)
    xticks.append(tick)

    tick += 2
    w_w2_x.append(tick)
    xticks.append(tick)


    tick += 4
fig1, ax1 = plt.subplots()
# ax1.margins(x=0, y=-.4)
ax1.set_ylabel('Runtime MAE ')
ax1.set_xlabel('Size of Original Data Used')
ax1.set_title("Predicted Runtime MAE on Tseitin Data")
ax1.set_ylim( 0, 1000)

# ax1.set_title("Original (Green) and Augmented (Blue) Predicted Runtime MAE on Tseitin Data")
ax1.set_xticklabels(xlabels, rotation=75)

# ax1.boxplot(sidebyside, sym='', patch_artist=True, positions=xlabels)
# xticks = xticks[:-2]

bplot = ax1.boxplot(sidebyside, sym='', patch_artist=True, positions=xticks, widths=[1.5]*len(sidebyside))

green = 0
color = ''
for patch in bplot['boxes']:
    if green % 5 == 0:
        color = 'green'
    if green % 5 == 1:
        color = 'blue'
    if green % 5 == 2:
        color = 'red'
    if green % 5 == 3:
        color = 'purple'
    if green % 5 == 4:
        color = 'brown'
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    green += 1

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='green', lw=2),
                Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='purple', lw=2)
                , Line2D([0], [0], color='brown', lw=2)
                ]
# ax1.legend(custom_lines, ['Original', 'HardCore', 'HardSATGEN-50', 'HardSATGEN-Strict', 'W2SAT'],loc='center left', bbox_to_anchor=(0.75, 0.75))
# ax1.legend(custom_lines, ['Original', 'HardCore', 'HardSATGEN-25', 'HardSATGEN-Strict', 'W2SAT'])
bplot_y = [item.get_ydata() for item in bplot['whiskers']]
w_og_y = []
w_hs_y = []
w_hss_y = []
w_w2_y = []
for i in range(len(x)):
    w_og_y.append(5  + bplot_y[i*10 + 1][1])
    w_hs_y.append(5 + bplot_y[i*10 + 5][1])
    w_hss_y.append(5 + bplot_y[i*10 + 7][1])
    w_w2_y.append(5 + bplot_y[i*10 + 9][1])
for i in range(len(w_og_x)):
    ax1.text(w_og_x[i], w_og_y[i], 'p ' + str(w_og[i][1]), ha = 'center', va = 'bottom', rotation=90, font={'size': 7})
    ax1.text(w_hs_x[i], w_hs_y[i], 'p ' + str(w_hs[i][1]), ha = 'center', va = 'bottom', rotation=90, font={'size': 7})
    ax1.text(w_hss_x[i], w_hss_y[i], 'p ' + str(w_hss[i][1]), ha = 'center', va = 'bottom', rotation=90, font={'size': 7})
    ax1.text(w_w2_x[i], w_w2_y[i], 'p ' + str(w_w2[i][1]), ha = 'center', va = 'bottom', rotation=90, font={'size': 7})

ax1.set_xlim(-3, tick)
ax1.set_ylim(None, np.max(np.stack(bplot_y))+25)

sidebyside_bars = []
for i in range(len(combined_trials)):
    sidebyside_bars.append(np.mean(original_trials[i]))
    sidebyside_bars.append(np.mean(combined_trials[i]))
    sidebyside_bars.append(np.mean(hardsatgen_trials[i]))
    sidebyside_bars.append(np.mean(hsStrict_trials[i]))
    sidebyside_bars.append(np.mean(w2_trials[i]))

fig2, ax1 = plt.subplots()
bar_xs = []
bar_x = 0
for i in range(len(combined_trials)):
    for j in range(5):
        bar_xs.append(bar_x)
        bar_x += 5
    
    bar_x += 10
barplot = ax1.bar(x=bar_xs, height=sidebyside_bars)