import os
import numpy as np
import shutil

cnf_dir = '/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new_nt/'
dst_dir = '/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new_nt_v6/'
os.mkdir(dst_dir)
csv_dir = '/home/joseph-c/sat_gen/cnfgen/tseitin_formulas_new_nt_csvs/runtimes_completed.csv'

# output_dir = '/home/joseph-c/sat_gen/qc_u15_output'
writestr = ''
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)
csv = open(csv_dir, 'r')
for line in csv.readlines():
    name = line.split(',')[0]
    times = line.split(',')[1].split('[')[1].split(']')[0].split(' ')
    realtimes = []
    for time in times:
        if time != '':
            realtimes.append(float(time))
    
    # if np.sum(realtimes) > 100 and os.path.exists(cnf_dir + '/' + name[:-4] + '_core'):
    # if np.sum(realtimes) > 1 and np.sum(realtimes) < 4500 and len(realtimes) == 7:
    realtimes_sorted = np.sort(realtimes) 
    if realtimes_sorted[0] != realtimes_sorted[1] and np.sum(realtimes) > 10:
        line = line.replace('1500.', '5000.')
        writestr += line
        shutil.copyfile(cnf_dir + '/' + name, dst_dir + '/' + name)
        # shutil.copyfile(cnf_dir + '/' + name[:-4] + '_core', dst_dir + '/' + name[:-4] + '_core')
if os.path.exists(dst_dir + '_csvs'):
    shutil.rmtree(dst_dir + '_csvs')
os.mkdir(dst_dir + '_csvs')
dst_csv = open(dst_dir + '_csvs' + '/' + 'runtimes_completed.csv', 'w')
# dst_csv_file = open(dst_csv, 'w')
dst_csv.write(writestr)



