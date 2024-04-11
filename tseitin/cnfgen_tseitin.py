import os
from multiprocessing import Pool
import random
import shutil
import cnfgen
import csv

# file = open('/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_nt_csvs/runtimes_completed.csv')
# reader = csv.reader(file)
# nt_args = []
# for row in reader:
#     nt_args.append(row[0].split('_')[1:3])
#     print(nt_args[-1])
# # exit()
args = []
filenames = {}
# for i in range(100):
for i in range(156250):
    for j in range(1):
        N = str(random.randrange(40 , 60))
        # d = str(random.uniform(0.05, 0.10))

        p = str(((int(N) - 40)*-0.004 + 0.11)*(1.0 + random.uniform(-0.05, 0.05))) #haven't tried with 0.002 yet, only 0.003
        # d = str(float(d)*(1.0 + random.uniform(-0.005, 0.005)))
        # d = str(random.uniform(0.05, 0.06))
        print(N, p)
        filename = 'tseitin_' + N + '_' + p
        if filename not in filenames.keys():
            filenames[filename] = 0
        filenames[filename] += 1
        filename += '_' + str(filenames[filename]) + '.cnf'

        
        args.append('cnfgen tseitin random gnp ' + N + ' ' + p + ' > /home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new/' + filename )

# exit()
pool = Pool(80)
pool.map(os.system, args)


pool.close()
pool.join()

# os.system(args[0])

# print('done')

for file in os.listdir('/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new/'):
    f = open('/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new/' + file)
    lines = f.readlines()
    f.close()
    if len(lines) < 2: 
        os.remove('/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new/' + file)

print('done')