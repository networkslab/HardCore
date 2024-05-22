import os
import csv
import numpy as np

csvs = '//home/joseph-c/sat_gen/tseitin_run_add_csvs/'
cnfdir = '/home/joseph-c/sat_gen/tseitin_run_add/'
f = open(csvs + 'fixed_runtimes.csv', 'a')
f.write(',name,#var,#clause,base,HyWalk,MOSS,mabgb,ESA,bulky,UCB,MIN' + '\n')

original_10 = {}
i = 0
with open(csvs + '/runtimes_completed.csv') as og:
    reader = csv.reader(og)
    
    counter = 0
    for row in reader:
        aasdf = row
        name = row[0]
        filename = name.split('.cnf_')[0].split('_post')[0]
        if row[1] == '[]': 
            continue
        row = row[1].replace("[", "").replace("]", "").replace(",", "").split()
        if filename not in original_10.keys():
            cnf_file = cnfdir + name
            try:
                # print('here here here')
                with open(cnf_file, 'r') as g:
                    # print('here here')
                    content = g.readlines()
                    # print('here')
                    while content[0].split()[0] == 'c':
                        content = content[1:]
                    while len(content[-1].split()) <= 1:
                        content = content[:-1]
            except:
                print(aasdf)
            # Parameters
            parameters = content[0].split()
           
            num_vars = int(parameters[2])
            num_clause = int(parameters[3])
            times = (list(map(float, row)))
            new_line = [i, name, num_vars, num_clause, times[0], times[5], times[1], times[4], times[3], times[6], times[2], np.min(times)]
            # times.append(np.min(times))
            # times.insert(0, i)
            # times.insert(1, name)
            # times.insert(2, num_vars)
            # times.insert(3, num_clause)
            write_str = ""
            for item in new_line:
                write_str += str(item) + ','
            write_str = write_str[:-1]
            f.write(write_str + '\n')
        else:
            print('uh oh')
        i += 1
            