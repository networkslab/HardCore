from pathlib import Path
import os
import random

import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
import dgl
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dglnn
# from model import SATInstanceEncoderHetero
import time
import sys
# from detect_core import detect_core
# from attach_new_lit import attach_new_lit

def to_int_matrix(formula, num_vars):
    rows = []
    cols = []
    appeared_var = []
    for _i in range(len(formula)):
        for ele in formula[_i].split()[: -1]:
            var = int(ele)
            if var > 0 :
                rows.append(var)
                appeared_var.append(var)
            else:
                rows.append(abs(var) + num_vars)
                appeared_var.append(abs(var))
            line_num = 2*num_vars + _i + 1
            cols.append(line_num)
    # add links between v and -v (only appeared variables)
    appeared_var = set(appeared_var)
    for var in appeared_var:
        rows.append(var)
        cols.append(var+num_vars)
    return rows, cols



def process_data(filename, core_file, data_dir, hetero_graphs):
    
    writepath = 'CoreDetection/trivial_data_gen/data'
    filename = filename + '.cnf'
    # print(filename, data_dir)
    # data_dir = args.data_dir
    cnfparse_start = time.time()
    # if os.path.isfile('/net/storage-1/home/j84299472/sat_gen/CoreDetection/trivial_data_gen/data/hetero_graphs' + '_' + filename.replace('/', '_') +'.pkl'):
    #     return
    cnf = open(data_dir+'/'+filename)
    cnf_content = cnf.readlines()
    while cnf_content[0].split()[0] == 'c':
        cnf_content = cnf_content[1:]
    while len(cnf_content[-1].split()) <= 1:
        cnf_content = cnf_content[:-1]
    
    parameters = cnf_content[0].split()
    cnf_formula = cnf_content[1:] # The clause part
   
    num_vars = int(parameters[2])
    num_clause = int(parameters[3])
    if not os.path.exists(data_dir +'/'+filename[:-4]+'_core'):
        return
    core = open(data_dir +'/'+filename[:-4]+'_core')
    # core = open(data_dir + '_core'+'/'+filename[:-4]+'_core')
    core_content = core.readlines()
    while core_content[0].split()[0] == 'c':
        core_content = core_content[1:]
    while len(cnf_content[-1].split()) <= 1:
        core_content = core_content[:-1]
    
    parameters = core_content[0].split()
    core_formula = core_content[1:] # The clause part
   
    num_vars = int(parameters[2])
    num_core_clause = int(parameters[3])
    
    labels = torch.zeros((num_clause,1))
    
    c = 0
    for origin_clause in cnf_formula:
        origin_flag = False
        for core_clause in core_formula:        
            if set(core_clause.split(' ')) == set(origin_clause.split(' ')):
                labels[c] = 1
                break
        c += 1
    if hetero_graphs is None:
    
        mat = to_int_matrix(cnf_formula, num_vars)
        rows = np.array(mat[0])
        cols = np.array(mat[1])
        rows_sym = np.concatenate((rows, cols))
        cols_sym = np.concatenate((cols, rows))
        n = num_vars * 2 + num_clause + 1

        mm = csr_matrix((np.ones(rows_sym.size, int), (rows_sym, cols_sym)), shape=(n, n))
        
        new_g = dgl.from_scipy(mm)
        
        # for var in range(num_vars):
        #     new_g.add_edges(1+var, 1+var + num_vars)
        new_g.add_edges(list(range(1, 1+num_vars)), list(range(1+num_vars,1+num_vars+num_vars)))
        new_g = dgl.remove_nodes(new_g, [0])
        node_types = [0]*num_vars + [1]*num_vars + [2]*num_clause
        # print(node_types)

        node_types_arr = np.array(node_types)
        edge_src = new_g.edges()[0].numpy()
        edge_dst = new_g.edges()[1].numpy()
        # edge_src = np.concatenate((edge_src, edge_dst))
        # edge_dst = np.concatenate((edge_dst, edge_src))

        src_type = node_types_arr[edge_src]
        dst_type = node_types_arr[edge_dst]

        edge_types = np.ones_like(src_type) * 2   # default edge_type = flip
        edge_types[dst_type==2] = 0  # edge_type = in
        edge_types[src_type==2] = 1  # edge_type = contain


        # edge_types = np.ones_like(src_type)   # default edge_type = flip
        # edge_types[dst_type!=2] = 0  # edge_type = in

        # edge_types = [0]*new_g.num_edges()
        
        new_g.ndata['_TYPE'] = torch.tensor(node_types)
        new_g.edata['_TYPE'] = torch.tensor(edge_types)
        # print(list(edge_types))

        # input()
        # print(len(list(edge_types)))
        # print(num_vars, num_clause)
        # print(new_g.all_edges())
        hg = dgl.to_heterogeneous(new_g, ['pos_lit','neg_lit','clause'], ['in', 'contain', 'flip'])
        # hg = dgl.to_heterogeneous(new_g, ['pos_lit','neg_lit','clause'], ['in', 'pair'])
        hetero_graphs=hg
    hg_info = [num_vars, num_clause]
    node_labels = labels
    # print(cnf_formula)
    cnfparse_end = time.time()
    # print(data_dir + '/' + filename + 'time spent parsing: ', cnfparse_end - cnfparse_start)
    save_name = data_dir.split('/')[-1] + '_' + filename
    pickle.dump(hg, open('/home/joseph-c/sat_gen/CoreDetection/trivial_data_gen/HardPSGEN/graphs/hetero_graphs' + '_' + save_name +'.pkl', 'wb'))
    pickle.dump([num_vars, num_clause], open('/home/joseph-c/sat_gen/CoreDetection/trivial_data_gen/HardPSGEN/graphs/hg_info' + '_' + save_name +'.pkl', 'wb'))
    pickle.dump(node_labels, open('/home/joseph-c/sat_gen/CoreDetection/trivial_data_gen/HardPSGEN/graphs/node_labels' + '_' + save_name +'.pkl', 'wb'))

def detect_core(filename, core_file, data_dir, hetero_graphs=None):
    process_data(filename, core_file, data_dir, hetero_graphs)
    # modelload_start = time.time()
    # saved_model = torch.load('/net/storage-1/home/j84299472/sat_gen/CoreDetection/HardPSGEN/scripts/model_1694112802.2163734', map_location=torch.device('cpu'))
    # saved_model.eval()
    # modelload_end = time.time()
    # print(data_dir + '/' + filename + 'time spent loading model: ', modelload_end - modelload_start)
    # modelrun_start = time.time()
    # num_vars, num_clause = hg_info
    # h_in = {'pos_lit': torch.ones((num_vars, 3)),
    #     'neg_lit': torch.ones((num_vars, 3)),
    #     'clause': torch.ones((num_clause, 3))}
    
    # # core_labels = saved_model(hetero_graphs, h_in)
    # core_labels = torch.ones(num_clause)
    # modelrun_end = time.time()
    # print(data_dir + '/' + filename + 'time spent running model: ', modelrun_end - modelrun_start)
    # # print(core_labels)
    # coresave_start = time.time()
    # core_formula = []
    # num_core_clause = np.sum(np.where(core_labels > 0.5, 1, 0))
    # with open(core_file, 'a') as cf:
    #     cf.write('c generated by me\n')
    #     cf.write('p cnf ' + str(num_vars)+ ' '+ str(num_core_clause) +'\n')
    #     for i in range(len(core_labels)):
    #         if core_labels[i] > 0.5 :
    #             cf.write(cnf_formula[i])
    #             # print('+1 clause!')
    #     # pickle.dump(core_formula, open(core_file, 'wb'))
    # # print('saved core to', core_file)
    # coresave_end = time.time()
    # print(data_dir + '/' + filename + data_dir + '/' + filename + 'time spent saving core: ', coresave_end - coresave_start)
    # return hetero_graphs

def read_files(cnf_file, core_file, save_file, origin_core, add_var=False, drat=None):
    with open(cnf_file) as cnf:
        content = cnf.readlines()
        while content[0].split()[0] == 'c':
            content = content[1:]
        num_vars = int(content[0].split(' ')[2])
        while content[0].split()[0] == 'p':
            content = content[1:]
        while len(content[-1].split()) <= 1:
            content = content[:-1]
    
    with open(core_file, encoding='windows-1252') as core:
        core_content = core.readlines()
        while core_content[0].split()[0] == 'c' or core_content[0].split()[0] == 'p':
            core_content = core_content[1:]
        while len(core_content[-1].split()) <= 1:
            core_content = core_content[:-1]
    
    if origin_core == None:
        return content, num_vars, core_content, None, None
    
    with open(origin_core) as origin:
        origin_content = origin.readlines()
        while origin_content[0].split()[0] == 'c' or origin_content[0].split()[0] == 'p':
            origin_content = origin_content[1:]
        while len(origin_content[-1].split()) <= 1:
            origin_content = origin_content[:-1]

    if drat == None:
        return content, num_vars, core_content, origin_content, None
    
    with open(drat) as drat:
        drat_content = drat.readlines()
        drat_content = drat_content[:-1]
    
    return content, num_vars, core_content, origin_content, drat_content


def attach_new_lit(cnf_file, core_file, save_file, origin_core, cnf_graph, add_var=False):
    content, num_vars, core_content, origin_content, _ = read_files(cnf_file, core_file, save_file, origin_core, add_var=add_var)

    random.shuffle(core_content)
    for core_clause in core_content:
        origin_flag = False
        for origin_clause in origin_content:
            if set(core_clause.split(' ')) == set(origin_clause.split(' ')):
                origin_content.remove(origin_clause)
                origin_flag = True
                break
        if origin_flag:
            # print('====================================================================================')
            continue
        
        origin_flag = False
        for idx, clause in enumerate(content):
            if set(core_clause.split(' ')) == set(clause.split(' ')):
                if add_var:
                    num_vars += 1
                    cnf_graph = dgl.add_nodes(cnf_graph, 1, ntype='pos_lit')
                    cnf_graph.nodes['pos_lit'][0]['_ID'][-1] = cnf_graph.num_nodes()
                    cnf_graph=dgl.add_nodes(cnf_graph, 1, ntype = 'neg_lit')
                    cnf_graph.nodes['neg_lit'][0]['_ID'][-1] = cnf_graph.num_nodes()
                    cnf_graph.add_edges(cnf_graph.num_nodes()-1, cnf_graph.num_nodes()-2, etype=('neg_lit', 'flip', 'pos_lit'))
                    cnf_graph.add_edges(cnf_graph.num_nodes()-2, cnf_graph.num_nodes()-1, etype=('pos_lit', 'flip', 'neg_lit'))
                     
                clause = f"{num_vars} " + clause
                cnf_graph.add_edges(idx+(num_vars-1)*2, cnf_graph.num_nodes()-2, etype = ('clause', 'contain', 'pos_lit'))
                cnf_graph.add_edges(cnf_graph.num_nodes()-2, idx+(num_vars-1)*2, etype = ('pos_lit', 'in', 'clause'))
                content[idx] = clause
                origin_flag = True
                break
        if origin_flag:
            break
    
    with open(save_file, 'w') as out_file:
        # print("writing to", args.save)
        out_file.write("c generated by G2SAT lcg\n")
        out_file.write("p cnf {} {}\n".format(num_vars, len(content)))
        for clause in content:
            out_file.write(clause)

    return
def process(args):

    no=0
    no1=1

    cnf_file=args[1]
    origin_core=args[2]
    num_iter=args[3]
    goal_time=args[4]
    class_name=args[5]
    data_dir=args[7]
    # print('cnf_file', cnf_file)
    # print('origin_core', origin_core)
    # print('data_dir', data_dir)
    # exit()
    

    cnf_name=cnf_file.split('/')[-1][:-4]
    

    # save_path=${cnf_file%/*}_post
    save_path = str(Path(cnf_file).parent.absolute()) + '_post'
    # print(save_path)
    # print(cnf_file)
    # exit()
    try:
        os.mkdir(save_path)
    except:
        save_path=save_path

    # drat_file=${cnf_file%.cnf}.drat
    # core_file=${cnf_file%.cnf}_core
    core_file = cnf_file[:-4] + '_core'
    # save_file=$save_path/${cnf_name}_r$no1.cnf
    save_file = save_path + '/' + cnf_name + '_r'+ str(no1) + '.cnf'
    home='/home/joseph-c/sat_gen/CoreDetection/HardPSGEN'

    # cd $home/src/postprocess/cadical/build
    # timeout 2500 ./cadical $cnf_file --no-binary $drat_file
    # cd $home/src/postprocess/drat-trim
    # timeout 2500 ./drat-trim $cnf_file $drat_file -c $core_file


    # cd $home/src/
    # python postprocess/learned_core_detection.py --filename $cnf_name  --core_file $core_file --data_dir "$data_dir"
    # python postprocess/sat_dataprocess.py --cnf $cnf_file --core $core_file --save $save_file --origin $origin_core --add_var
    hg = detect_core(cnf_name, core_file, data_dir)
    # hg = attach_new_lit(cnf_file, core_file, save_file, origin_core, hg, add_var=True)
    
    # # rm $core_file
    # os.remove(core_file)
    # # cd ..

    # # while [ "$no" -lt $num_iter ]; do
    # while no < int(num_iter):
    #     # no=$((no + 1))
    #     no = no + 1
    #     # no1=$((no1 + 1))
    #     no1 = no1 + 1
        
    #     # cnf_file=$save_file
    #     cnf_file = save_file
    #     # drat_file=${cnf_file%.cnf}.drat
    #     # core_file=${cnf_file%.cnf}_core
    #     core_file = cnf_file[:-4] + '_core'
    #     # save_file=$save_path/${cnf_name}_r$no1.cnf
    #     save_file = save_path + '/' + cnf_name + '_r' + str(no1) + '.cnf'

    #     # cd $home/src/postprocess/cadical/build
    #     # timeout 3000 ./cadical $cnf_file --no-binary $drat_file &>> $home/src/log/${class_name}_${cnf_name%.cnf}.log
    #     # time=`tail -n 7 $home/src/log/${class_name}_${cnf_name%.cnf}.log`
    #     # time=${time#*:}
    #     # time=${time%%seconds*}
    #     # time=`eval echo $time|awk '{print $1}'`
        
        

    #     # cd $home/src/postprocess/drat-trim
    #     # timeout 3000 ./drat-trim $cnf_file $drat_file -c $core_file
    #     # cd $home/src
    #     # python postprocess/learned_core_detection.py --core_file $core_file --filename $cnf_name --data_dir "$data_dir" 
    #     # python postprocess/sat_dataprocess.py --cnf $cnf_file --core $core_file --save $save_file --origin $origin_core
    #     hg = detect_core(cnf_name, core_file, data_dir, hetero_graphs=hg)
    #     try:
    #         hg = attach_new_lit(cnf_file, core_file, save_file, origin_core, hg, add_var=False)
    #     except:
    #         print('breaking for ', save_file)
    #         break
    #     # if [ `echo "$time > $goal_time" | bc` -eq 1 ]
    #     # then 
    #     #     # rm $drat_file
    #     #     echo breaking for time $goal_time 
    #     #     break 1
    #     # fi
    #     #if savefile is file
        
    #     # if  [ ! -f $savefile ]; then
        
    #     #     rm $core_file
    #     #     # rm $drat_file
    #     #     cd ..
    #     #     echo breaking for $save_file
    #     #     break 1
            
        
    #     # fi
    #     os.remove(cnf_file)
    #     os.remove(core_file)
    #     # rm $cnf_file
    #     # rm $core_file

    #     #   rm $drat_file  
    #     # cd ..
        


import os

import csv


       

if __name__ == "__main__":
    n_gen_dir = '/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/dataset/tseitin_formulas_nt/n_gen_clause.csv'
    # n_gen_dir = '/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/dataset/lec/n_gen_clause.csv'
    original = {}
    counter = 0
    # with open(n_gen_dir) as og:
    #     reader = csv.reader(og)
    # for row in reader:
    #     name = row[0]
    #     n_gen = row[1]
    #     # original[name] = str(n_gen)
    #     if int(n_gen) > 200:
    #         original[name] = [str(200), 1000]
    #     else:
    #         original[name] = [str(n_gen), 1000]
    for file in os.listdir('/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/dataset/tseitin_formulas_nt'):
        if file[-4:] == 'core':
            key = file[:-5] + '.cnf'
            original[key] = [str(200), 1000]
    log_file = open('/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/scripts/log/PS_generated/runtimes.log', 'w')
    # sys.stdout = log_file
    counter = 0
    # with open("/net/storage-1/home/j84299472/sat_gen/CoreDetection/lec_runtime_csvs/runtimes_completed.csv") as og:
    # # with open("/home/joseph-c/sat_gen/lec_runtime_csvs/runtimes_completed.csv") as og:
    #     reader = csv.reader(og)
    #     for row in reader:
    #         name = row[0]
    #         base_time = row[1].replace("[", "").replace("]", "").replace(",", "").split()[0]
    #         base_time = float(base_time)*1.25
    #         base_time = int(base_time)
    #         # print(base_time)
    #         if name not in original.keys():
    #             print('deleted ', name)
    #             continue
    #         else:
    #             original[name] = [original[name], str(base_time)]
            
    args_list = []
    home = '/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/'

    for filename in os.listdir('/home/joseph-c/sat_gen/CoreDetection/trivial_data_gen/HardPSGEN/formulas/tseitin_train_post'):
        # key = key[:-4]
        # f = os.path.join(home +'/formulas/PS_generated/' + key)
        f = os.path.join('/home/joseph-c/sat_gen/CoreDetection/trivial_data_gen/HardPSGEN/formulas/tseitin_train_post', filename)
        # checking if it is a file
        if os.path.isfile(f) and filename[-3:] == 'cnf':
            input = f
            core_input = f[:-4] + '_core'
            value = [None, None]
            num_iter = value[0]
            goal_time = value[1]
            class_name = 'tseitin_train_post'
            cnf_name = filename[:-3]
            data_dir = '/home/joseph-c/sat_gen/CoreDetection/trivial_data_gen/HardPSGEN/formulas/tseitin_train_post'
            # device = gpus[counter]
            args_list.append(['blank', input, core_input, num_iter, goal_time, class_name, cnf_name, data_dir])
        if not os.path.isfile(f):
            print(f)
    # for key, value in original.items():
    #     key = key[:-4]
    #     if os.path.exists(home +'formulas/PS_generated/' + key ):
    #         for filename in os.listdir(home+'formulas/PS_generated/' + key):
    #         # for filename in os.listdir(home):
    #             f = os.path.join(home +'/formulas/PS_generated/' + key, filename,)
    #             # checking if it is a file
    #             if os.path.isfile(f) and filename[-3:] == 'cnf':
    #                 input = f
    #                 core_input = home+'formulas/PS_generated/' + key + '_core/' + filename
    #                 num_iter = value[0]
    #                 goal_time = value[1]
    #                 class_name = 'PS_generated/' + key
    #                 cnf_name = filename[:-3]
    #                 data_dir = home+'formulas/PS_generated/' + key 
    #                 # device = gpus[counter]
    #                 args_list.append(['blank', input, core_input, num_iter, goal_time, class_name, cnf_name, data_dir])
    #             if not os.path.isfile(f):
    #                 print(f)
        # else:
            # print(home +'formulas/PS_generated/' + key )
    # print(original.items())

    
    from multiprocessing import Pool
    # for i in range(1+int(len(args_list)/1000)):
    #     if (i+1)*1000 < len(args_list):
    #         pool = Pool(1)
    #         #print(len(paths))
    #         pool.map(run_gen, args_list[i*1000:(i+1)*1000])

    #         pool.close()
    #         pool.join()
            
    #     else:
    #         pool = Pool(200)

    #         pool.map(run_gen, args_list[i*1000:])

    #         pool.close()
    #         pool.join()
    # print('data_dir', args_list[0][6])
    # process(args_l5AF3pa55w0ist[0])

    # exit()



    pool = Pool(100)
    #print(len(paths))
    pool.map(process, args_list)

    pool.close()
    pool.join()
    log_file.close()
    # for arg in args_list:
    #     run_gen(arg)
    # print()