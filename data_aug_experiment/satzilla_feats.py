import os
print(os.getcwd())
import pandas as pd
from tqdm import tqdm
import numpy as np
import dgl
import torch
import pickle as pkl
import argparse
from time import time
# from multiprocessing import Pool
# from dgl.data.utils import save_graphs
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

def func_to_int_matrix(formula, num_vars, num_clause):
    # all idx of literals&clause minus 1, delete the "0" in the CNF
    edges = []
    occ_for_vars = torch.zeros((num_vars, 2))  # Count the positive and negative occurances of each variable
    occ_in_clause = torch.zeros((num_clause, 2)) # Count the positive and negative leteriasl in each clause
    occ_in_horn = torch.zeros((num_vars))
    num_horn_clause = 0
    for _i in range(len(formula)):
        for ele in formula[_i].split()[: -1]:
            var = int(ele)
            var_id = abs(var) - 1
            clause_id = num_vars + _i
            if var > 0 :
                occ_in_clause[_i, 0] += 1
                occ_for_vars[var_id, 0] += 1
            else:
                occ_in_clause[_i, 1] += 1
                occ_for_vars[var_id, 1] += 1
            edges.append([var_id, clause_id])
        # Horn clause (with at most one positive literal)
        if occ_in_clause[_i, 0] < 2: 
            num_horn_clause += 1
            for ele in formula[_i].split()[: -1]:
                var = int(ele)
                var_id = abs(var) - 1
                occ_in_horn[var_id] += 1

    rows = np.array(edges)[:,0]
    cols = np.array(edges)[:,1]
    frac_horn_clause = num_horn_clause / num_clause
    return rows, cols, occ_in_clause, occ_for_vars, frac_horn_clause, occ_in_horn

def calc_entropy(data):
    _, val_cnt = torch.unique(data, return_counts=True)
    p = val_cnt / val_cnt.sum()
    entr = -torch.sum(p*torch.log(p))
    return entr

# ========================== get train sample (combine) ============================================
def get_satzilla_features(line):
    #print(df)
    #key = list(df['name'])[i]  # start: name = '0'
    t0 = time()
    cnf = open(line.strip('\n'), 'r')
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Parameters
    parameters = content[0].split()
    formula = content[1:] # The clause part of the dimacs file

    all_feats = []
    feat_names = []

    # 1. Number of Clauses
    num_clause = int(parameters[3])
    # 2. Number of Variables
    num_vars = int(parameters[2])
    # 3. Ratio: c/v
    ratio_cv = num_clause / num_vars

    all_feats.extend([num_clause, num_vars, ratio_cv])
    feat_names.extend(['num_clause', 'num_vars', 'ratio_cv'])

    # Convert to variable-clause graph to get degrees info
    rows, cols, occ_in_clause, occ_for_vars, frac_horn_clause, occ_in_horn = func_to_int_matrix(formula, num_vars, num_clause)

    # num_literals = num_vars * 2
    # n = num_literals + num_clause
    num_nodes = num_vars + num_clause

    g = dgl.graph((rows, cols), num_nodes=num_nodes, idtype=torch.int32)
    bg = dgl.to_bidirected(g)

    degs = (g.in_degrees() + g.out_degrees()).float()

    # 4-8. Variable nodes degree statistics: mean, variation coefficient, min, max and entropy
    var_degs = degs[:num_vars]
    var_degs_mean = torch.mean(var_degs)
    var_degs_var = torch.var(var_degs)
    var_degs_min = torch.min(var_degs)
    var_degs_max = torch.max(var_degs)
    var_degs_entr = calc_entropy(var_degs)

    all_feats.extend([var_degs_mean, var_degs_var, var_degs_min, var_degs_max, var_degs_entr])
    feat_names.extend(['var_degs_mean', 'var_degs_var', 'var_degs_min', 'var_degs_max', 'var_degs_entr'])

    # 9-13. Clause nodes degree statistics: mean, variation coefficient, min, max and entropy
    clause_degs = degs[num_vars:]
    clause_degs_mean = torch.mean(clause_degs)
    clause_degs_var = torch.var(clause_degs)
    clause_degs_min = torch.min(clause_degs)
    clause_degs_max = torch.max(clause_degs)
    clause_degs_entr = calc_entropy(clause_degs)

    all_feats.extend([clause_degs_mean, clause_degs_var, clause_degs_min, clause_degs_max, clause_degs_entr])
    feat_names.extend(['clause_degs_mean', 'clause_degs_var', 'clause_degs_min', 'clause_degs_max', 'clause_degs_entr'])

    # 14-17. Nodes degree statistics: mean, variation coefficient, min, max and entropy
    degs_mean = torch.mean(degs)
    degs_var = torch.var(degs)
    degs_min = torch.min(degs)
    degs_max = torch.max(degs)
    # clause_degs_entr = calc_entropy(clause_degs)
    all_feats.extend([degs_mean, degs_var, degs_min, degs_max])
    feat_names.extend(['degs_mean', 'degs_var', 'degs_min', 'degs_max'])

    # 18-20. Ratio of positive and negative literals in each clause
    ratio_in_clause = occ_in_clause[:, 0] / occ_in_clause.sum(dim=1)
    ril_mean = torch.mean(ratio_in_clause)
    ril_var = torch.var(ratio_in_clause) 
    ril_entr = calc_entropy(ratio_in_clause)

    all_feats.extend([ril_mean, ril_var, ril_entr])
    feat_names.extend(['ril_mean', 'ril_var', 'ril_entr'])
    
    # 21-25. Ratio of positive and negative occurances of each variables
    ratio_literal_sign = occ_for_vars[:, 0] / occ_for_vars.sum(dim=1)
    ratio_literal_sign = ratio_literal_sign.nan_to_num(nan=0.0)
    rls_mean = torch.mean(ratio_literal_sign)
    rls_var = torch.var(ratio_literal_sign)
    rls_min = torch.min(ratio_literal_sign)
    rls_max = torch.max(ratio_literal_sign)
    rls_entr = calc_entropy(ratio_literal_sign)

    all_feats.extend([rls_mean, rls_var, rls_min, rls_max, rls_entr])
    feat_names.extend(['rls_mean', 'rls_var', 'rls_min', 'rls_max', 'rls_entr'])
    
    # 26-27. Fraction of binary and ternary clause
    frac_binary = (clause_degs==2).sum() / num_clause 
    frac_ternary = (clause_degs==3).sum() / num_clause 

    all_feats.extend([frac_binary, frac_ternary])
    feat_names.extend(['frac_binary', 'frac_ternary'])

    # 28. Fraction of horn clauses
    frac_horn_clause = frac_horn_clause # Computed above

    all_feats.append(frac_horn_clause)
    feat_names.append('frac_horn_clause')

    # 29-33. Number of occurence in a Horn clause for each variable
    horn_mean = torch.mean(occ_in_horn)
    horn_var = torch.var(occ_in_horn)
    horn_min = torch.min(occ_in_horn)
    horn_max = torch.max(occ_in_horn)
    horn_entr = calc_entropy(occ_in_horn)

    all_feats.extend([horn_mean, horn_var, horn_min, horn_max, horn_entr])
    feat_names.extend(['horn_mean', 'horn_var', 'horn_min', 'horn_max', 'horn_entr'])

    satzilla_feats = torch.tensor(all_feats)

    process_time = time() - t0
    time_to_save = {'num_var': num_vars, 'num_clause': num_clause, 'num_nodes': num_nodes, 'process_time': process_time }
    return bg, time_to_save, satzilla_feats, feat_names

def add_satzilla_features():
    # graph_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset_u20k/sc_data_u20k_2254.bin'
    # info_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset_u20k/sc_data_u20k_2254_info.pkl'
    cnf_dir = '/home/joseph-c/sat_gen/HardSATGEN/formulas/2k_bc_post/'
    save_dir = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/'
    csv_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/2k_bc_post_hardsatgen_csvs/runtimes.csv'
    import csv

    csv_file = open(csv_path)
    csv_reader = csv.reader(csv_file)
    # graphs, label_dict = load_graphs(graph_path)
    # labels = label_dict['label']
    cnf_names = []
    # for filename in os.listdir(cnf_dir):
    #     if filename[-3:] == 'cnf':
    #         cnf_names.append(filename)
    first_line = True
    for row in csv_reader:
        if first_line:
            first_line = False
            continue
        # if '_post' in row[1]:
        #     continue
        cnf_names.append(row[1])
    # cnf_names = ['weishan-_-weishan_big_B035-_-TC2-_-DYEQ_BACK_P7-_-RTL2SYN33405_274630.cnf']
    # df = pd.read_csv('data/cnf_results.csv', delimiter=',')
    # df_rt = pd.DataFrame(df, columns = ['base', 'bulky', 'HyWalk', 'MOSS', 'ESA', 'mabgb', 'UCB'])

    # read_file = open('data/cnfpath_file.txt', 'r')
    # lines = read_file.readlines()

    # lines = lines[:1000]
    # with Pool(processes=args.n_pool) as pool:
    #     graphs = list(tqdm(pool.imap(cnf_to_dgl, lines), total=len(lines)))
    new_graphs = []
    all_process_time = []
    all_feats = []
    # num_samples = 10000
    # lines = lines[:num_samples]
    for cnf_name in tqdm(cnf_names):
        cnf_file = os.path.join(cnf_dir, cnf_name)
        _, to_save, feats, feat_names = get_satzilla_features(cnf_file)
        # g.ndata['satzilla_feats'] = feats

        to_save.update({'name': cnf_name})
        all_process_time.append(to_save)
        all_feats.append(feats)

        # if len(all_process_time) == 3:
        #     break

    process_time = pd.DataFrame.from_records(all_process_time)
    process_time.to_csv(os.path.join(save_dir, 'satzilla_processing_time.csv'))

    # new_graph_path = os.path.join(save_dir, 'sc_data_u20k_2254_satzilla.bin')
    # save_graphs(new_graph_path, graphs, {'label': labels})
    
    satzilla_feats = torch.stack(all_feats).numpy()
    satzilla_df = pd.DataFrame(data=satzilla_feats, index=process_time['name'], columns=feat_names)
    satzilla_df.to_csv(os.path.join(save_dir, 'satzilla_feats_test.csv'))

    # graph_labels = {"glabel": torch.tensor(labels_orig[:, 0].tolist())}
    # save_graphs("processed/sat_data_satzilla.bin", graphs, graph_labels)

if __name__ == '__main__':
    add_satzilla_features()
