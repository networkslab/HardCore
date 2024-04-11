# from utils import *
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from tqdm import tqdm

data_path = '/home/vincent/sat/sat_selection/data/'
curr_dir = os.path.dirname(__file__)

# ========================================== Build Sparse Adjacent Matrix ==========================================
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
# ========================================== Build Sparse Adjacent Matrix End ==========================================


def process_data():
    cnf_dir = os.path.join(data_path, 'cnfs_1000')
    label_filename = os.path.join(data_path, 'cnfs_1000_results.csv')
    df = pd.read_csv(label_filename)

    store_dir = os.path.join(curr_dir, 'mtx')
    Path(store_dir).mkdir(parents=True, exist_ok=True)

    num_cnfs = df.shape[0]
    for i in tqdm(range(num_cnfs)):
        cnf_name = df['name'][i]
        cnf_file = os.path.join(cnf_dir, cnf_name)

        with open(cnf_file, 'r') as f:
            content = f.readlines()

        while content[0].split()[0] == 'c':
            content = content[1:]
        while len(content[-1].split()) <= 1:
            content = content[:-1]

        # Parameters
        parameters = content[0].split()
        formula = content[1:] # The clause part of the dimacs file
        _num_vars = int(parameters[2])
        _num_clause = int(parameters[3])
        formula = to_int_matrix(formula, _num_vars)
        rows = np.array(formula[0])
        cols = np.array(formula[1])

        n = _num_vars * 2 + _num_clause + 1
        mm = csr_matrix((np.ones(rows.size, int), (rows, cols)), shape=(n, n))
        # print("sparse matrix shape:", mm.shape)
        mtx_filename = store_dir + '/' + cnf_name.split('.')[0] + '.mtx'
        mmwrite(mtx_filename, mm)

if __name__ == '__main__':
    process_data()