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
from model import SATInstanceEncoderHetero
import time

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

def make_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--core_file', type=str)
    parser.add_argument('--data_dir', type=str)
    
    return parser

def process_data(args):
    
    writepath = 'CoreDetection/trivial_data_gen/data'
    filename = args.filename + '.cnf'
    data_dir = args.data_dir
    cnfparse_start = time.time()
    if os.path.isfile('/net/storage-1/home/j84299472/sat_gen/CoreDetection/trivial_data_gen/data/hetero_graphs' + '_' + filename.replace('/', '_') +'.pkl'):
        return
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
    # core = open(data_dir +'/'+filename[:-4]+'_core')
    # # core = open(data_dir + '_core'+'/'+filename[:-4]+'_core')
    # core_content = core.readlines()
    # while core_content[0].split()[0] == 'c':
    #     core_content = core_content[1:]
    # while len(cnf_content[-1].split()) <= 1:
    #     core_content = core_content[:-1]
    
    # parameters = core_content[0].split()
    # core_formula = core_content[1:] # The clause part
   
    num_vars = int(parameters[2])
    # num_core_clause = int(parameters[3])
    
    # labels = torch.zeros((num_clause,1))
    
    # c = 0
    # for origin_clause in cnf_formula:
    #     origin_flag = False
    #     for core_clause in core_formula:        
    #         if set(core_clause.split(' ')) == set(origin_clause.split(' ')):
    #             labels[c] = 1
    #             break
    #     c += 1
    
    
    mat = to_int_matrix(cnf_formula, num_vars)
    rows = np.array(mat[0])
    cols = np.array(mat[1])
    rows_sym = np.concatenate((rows, cols))
    cols_sym = np.concatenate((cols, rows))
    n = num_vars * 2 + num_clause + 1

    mm = csr_matrix((np.ones(rows_sym.size, int), (rows_sym, cols_sym)), shape=(n, n))
    
    new_g = dgl.from_scipy(mm)
    
    for var in range(num_vars):
        new_g.add_edges(1+var, 1+var + num_vars)
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
    # node_labels = labels
    # print(cnf_formula)
    cnfparse_end = time.time()
    print('time spent parsing: ', cnfparse_end - cnfparse_start)
    return hetero_graphs, hg_info, cnf_formula

def detect_core(args):
    hetero_graphs, hg_info, cnf_formula = process_data(args)
    modelload_start = time.time()
    saved_model = torch.load('/net/storage-1/home/j84299472/sat_gen/CoreDetection/HardPSGEN/src/postprocess/model_1694112802.2163734', map_location=torch.device('cpu'))
    saved_model.eval()
    modelload_end = time.time()
    print('time spent loading model: ', modelload_end - modelload_start)
    modelrun_start = time.time()
    num_vars, num_clause = hg_info
    h_in = {'pos_lit': torch.ones((num_vars, 3)),
        'neg_lit': torch.ones((num_vars, 3)),
        'clause': torch.ones((num_clause, 3))}
    
    core_labels = saved_model(hetero_graphs, h_in)
    modelrun_end = time.time()
    print('time spent running model: ', modelrun_end - modelrun_start)
    # print(core_labels)
    coresave_start = time.time()
    core_formula = []
    core_file = args.core_file
    num_core_clause = np.sum(np.where(core_labels > 0.5, 1, 0))
    with open(core_file, 'a') as cf:
        cf.write('c generated by me\n')
        cf.write('p cnf ' + str(num_vars)+ ' '+ str(num_core_clause) +'\n')
        for i in range(len(core_labels)):
            if core_labels[i] > 0.5 :
                cf.write(cnf_formula[i])
                # print('+1 clause!')
        # pickle.dump(core_formula, open(core_file, 'wb'))
    # print('saved core to', core_file)
    coresave_end = time.time()
    print('time spent saving core: ', coresave_end - coresave_start)

if __name__ == "__main__":
    args = make_arguments().parse_args()
    detect_core(args)
    