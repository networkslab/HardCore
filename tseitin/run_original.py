import os

# def run_dir (source_dir, output_dir):
#     executables= ["KISSAT-NEW/build/kissat", "moss/build/kissat/build/kissat",   "ucb/build/kissat/build/kissat", "esa/build/kissat", "mabgb/build/kissat/build/kissat", "hywalk/build/kissat","bulky/build/kissat/build/kissat"]       
#     log_files_path=source_dir
#     #log_files_path="data/CNFgen/$"
#     timeout_duration="1000s"
#     output_dir=output_dir

#     os.system('mkdir -p ' + output_dir)

#     def process_file(input):
#         log_file, output_file, index, executable = input
#         os.system('echo "Running ' + log_file + " with " + executable + " and saving output to " + output_file + '"')
#         os.system("timeout 5000 ./" + executable + " " + log_file + " --relaxed > " + output_file)
#         os.system('echo "Finished ' + output_file + '"')
#     args_list = []
#     for index in range(len(executables)):
#         for filename in os.listdir(log_files_path):
#             f = os.path.join(log_files_path, filename)
#             # checking if it is a file
#             if os.path.isfile(f) and filename[-3:] == 'cnf':
#                 log_file_name = filename
#                 output_file = output_dir + "/" + log_file_name + "_track_" + str(index) + ".log"
#                 args_list.append([f, output_file, index,executables[index]])
#     from multiprocessing import Pool
#     pool = Pool(170)
#     #print(len(paths))
#     pool.map(process_file, args_list)

#     pool.close()
#     pool.join()


log_files_path = '/home/joseph-c/sat_gen/tseitin_run_add/'
# gen_files_path = '/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/formulas/PS_generated/'
output_dir = '/home/joseph-c/sat_gen/tseitin_run_add_output/'
os.mkdir(output_dir)
# executables= ["KISSAT-NEW/build/kissat", "moss/build/kissat/build/kissat",   "ucb/build/kissat/build/kissat", "esa/build/kissat", "mabgb/build/kissat/build/kissat", "hywalk/build/kissat","bulky/build/kissat/build/kissat"]       
executables= ["KISSAT-NEW/build/kissat_slow", "moss/build/kissat/build/kissat_slow",   "ucb/build/kissat/build/kissat_slow", "esa/build/kissat_slow", "mabgb/build/kissat/build/kissat_slow", "hywalk/build/kissat","bulky/build/kissat/build/kissat_slow"]       


# executables= ["KISSAT-NEW/build/kissat"]
rungen = 0
#log_files_path="data/CNFgen/$"
timeout_duration="5000s"


# os.system('mkdir -p ' + output_dir)

def process_file(input):
    log_file, output_file, index, executable = input
    os.system('echo "Running ' + log_file + " with " + executable + " and saving output to " + output_file + '"')
    os.system("timeout 5000 ./" + executable + " " + log_file + " --relaxed > " + output_file)
    os.system('echo "Finished ' + output_file + '"')
args_list = []

for filename in os.listdir(log_files_path):
    # if filename[-3:] == 'cnf':
    if 1 > 0:
        if rungen == 1:
            f = os.path.join(log_files_path + filename)
            if os.path.exists(f):
                true_output = output_dir + filename

                # checking if it is a file

                for index in range(len(executables)):                    
                    output_file = true_output  + "_track_" + str(index) + ".log"
                    if os.path.isfile(output_file):
                        continue
                    else:
                        args_list.append([f, output_file, index,executables[index]])
        elif rungen == 0:
            f = os.path.join(log_files_path, filename)
            if os.path.exists(f):
                
                # checking if it is a file
                
                for index in range(len(executables)):
                    output_file = output_dir + filename + "_track_" + str(index) + '.log'
                    if os.path.isfile(output_file):
                        continue
                    else:
                        args_list.append([f, output_file, index,executables[index]])
    else:
        print(1)                
from multiprocessing import Pool
pool = Pool(100)
#print(len(paths))
pool.map(process_file, args_list)

pool.close()
pool.join()


        