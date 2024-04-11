import numpy as np
import pandas as pd
import pickle 

# test_idx_path = '/home/joseph-c/sat_gen/CNFgen/denoising/test_cherry_idx.npy'
# test_idx = np.load(test_idx_path)
# train_idx_path = '/home/joseph-c/sat_gen/CNFgen/denoising/train_cherry_idx.npy'
# train_idx = np.load(train_idx_path)

import csv

original = []
with open("/home/joseph-c/sat_gen/lec_runtime_csvs/" 'runtimes_completed.csv') as og:
    reader = csv.reader(og)
    
    counter = 0
    for row in reader:
        

        row = row[1].replace("[", "").replace("]", "").replace(",", "").split()
        original.append(list(map(float, row)))

o_counts = np.zeros(7)
for i in range(len(original)):
    o_counts[np.argmin(original[i])] += 1

print(o_counts)

exit()


shuffled = pd.read_csv('/home/joseph-c/sat_gen/CNFgen/denoising/cherry_par_2_shuffled.csv')
shuffled_by_name = pd.DataFrame(shuffled, index = shuffled['name'])
unshuffled = pd.read_csv('/home/joseph-c/sat_gen/CNFgen/denoising/cherry_par_2_unshuffled.csv')
original =  pd.read_csv('/home/joseph-c/sat_gen/minidata/cnfs_1000_results_0220.csv')
with open('/home/joseph-c/sat_gen/CNFgen/denoising/minidata_cherry_list.pkl', 'rb') as f:
     names = pickle.load(f)
counter = 0
s_counts = np.zeros(7)
o_counts = np.zeros(7)
u_counts = np.zeros(7)

# print(len(original))
# exit()
# for idx in test_idx[0]:
for idx in len():
    # print(shuffled.iloc[idx]['base'])
    # print(unshuffled.iloc[idx]['base']) 
    # names.append(shuffled.iloc[idx]['name'])
    s_counts[np.argmin(shuffled.iloc[idx][4:11])] += 1
    u_counts[np.argmin(unshuffled.iloc[idx][4:11])] += 1
    if shuffled.iloc[idx]['name'] not in names:
        print('uh oh')
procs = 0
for i in range(len(original)):
    if original.iloc[i]['name'] in names:
        o_counts[np.argmin(original.iloc[i][4:11])] += 1
# print(procs)
print('unshuffled', u_counts)
print('mean_shuffled', s_counts)
print('original', o_counts)
shuffle_5 = []
import csv
for i in  range(5):
    shuffle_5.append([])
    with open("/home/joseph-c/sat_gen/CNFgen/cherry_shuffle_runtime_csvs/" + str(i) + '_runtimes_completed.csv') as og:
        reader = csv.reader(og)
        
        counter = 0
        for row in reader:
            if i == 0:
                
                names.append([row[0]])
                file = open('/home/joseph-c/sat_gen/CNFgen/ShuffleCherry/Shuffle0_/' + names[-1][0])
                stats = file.readlines()[0].strip().split(" ")[2:]
                names[-1].append(int(stats[0]))
                names[-1].append(int(stats[1]))
            row = row[1].replace("[", "").replace("]", "").replace(",", "").split()
            shuffle_5[i].append(list(map(float, row)))
shuffle_5 = np.asarray(shuffle_5)
# print(shuffle_5.shape)
# exit()
si_counts = np.zeros((5, 7))
for i in range(5):
    for idx in train_idx[0]:
    # print(shuffled.iloc[idx]['base'])
    # print(unshuffled.iloc[idx]['base']) 
    # names.append(shuffled.iloc[idx]['name'])
        si_counts[i,np.argmin(shuffle_5[i,idx, :])] += 1
    print('shuffle number ', i, si_counts[i])

# print(unshuffled.iloc[1][4:11])
# print(shuffled.iloc[1][4:11])
# print(counter)
# print(test_idx[0])

