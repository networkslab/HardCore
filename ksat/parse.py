import numpy as np
import csv
import os
#for circuit in [ 22, 23, 24,26, 27, 28]
#for circuit in [ 23]:
    #for subdir in ["atmost2_1_", "atmost3_1_", "atmost4_3_", "equal2_", "exact2_1_", "exact2_2_", "exact3_3_", "exact4_4_", "xor2_"]:
    #for subdir in ["atmost2_1_"]:
directory = '/home/joseph-c/sat_gen/tseitin_run_add/'
# gen_files_path = '/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/formulas/PS_generated/'
log_directory = '/home/joseph-c/sat_gen/tseitin_run_add_output/'
csv_directory = '/home/joseph-c/sat_gen/tseitin_run_add_csvs/'
times_dict = {}
finished_dict = {}
os.mkdir(csv_directory)
    
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename[-3:] == 'cnf':
            
        time = []
        finished = []
        
        for i in [0, 1, 2, 3, 4, 5, 6]:
        
        
            g = os.path.join(log_directory, filename+"_track_"+str(i)+".log")
            
            if os.path.isfile(g):
                #print("isfile!")
                #print(g)
                lines = []
                start_line = 0
                line_counter = 0
                with open(g) as file:
                    for line in file:
                        
                        lines.append(line.strip())
                        if "resources" in line:
                            start_line = line_counter
                            #print(start_line)
                        line_counter += 1
                    
                
                if start_line == 0:
                    continue
                try:
                    #print(lines[start_line:])
                    #time.append(float(lines[-2].replace(" ", "").split('s')[3].split('seconds')[0]))
                    #print(g)
                    #print(start_line)
                    #print(lines[start_line+4])
                    #print(float(lines[start_line+3].split("seconds")[0].split(' ')[-2]))
                    time.append(float(lines[start_line+3].split("seconds")[0].split(' ')[-2]))
                    #print(int(lines[start_line+4].split("raising signal")[1].replace(" ", "")[0:2]))
                    finished.append(int(lines[start_line+4].split("raising signal")[1].replace(" ", "")[0:2]))
                    #print(time)
                except:
                    
                    finished.append(int(lines[-1][-2:]))
            else:
                print("not a file:",g)
        times_dict[filename] = time
        finished_dict[filename] = finished
        

with open(csv_directory +'/solver_times'+ '.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in times_dict.items():
        writer.writerow([key, value])
with open(csv_directory +'/solver_finished'+ '.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in finished_dict.items():
        writer.writerow([key, value])
run_completed = {}
for key, value in times_dict.items():
    run_completed[key] = np.where(np.asarray(finished_dict[key]) !=15,  times_dict[key], 1500)
with open(csv_directory +'/runtimes_completed'+ '.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in run_completed.items():
        writer.writerow([key, value])
