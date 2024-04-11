import csv
import numpy as np
import torch
from scipy import stats
import pandas as pd
import os
import argparse
def main(datasize):
    path = '/home/joseph-c/lightning_logs/0604_satzilla_mlp/sat_reg_unleaky/'
    outputs_path = path + '/test_pred_probs.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/bench_ranking_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/bench_ranking_strictAgain3_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og/bench_raking_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/bench_ranking_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/bench_ranking_v4_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_og/bench_ranking_v4_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/bench_ranking_v4_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/bench_ranking_loose_v4_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/bench_ranking_strict_v4_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/bench_ranking_strict_v5_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/bench_ranking_loose_v5_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/bench_ranking_v5_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/bench_ranking_v5_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/bench_ranking_v9_' + str(datasize) + '.csv'
    results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_og/bench_ranking_v9_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/bench_ranking_loose_v9_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/bench_ranking_strict_v9_' + str(datasize) + '.csv'
    # results_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/bench_ranking_v9_' + str(datasize) + '.csv'
    sep_results_path = results_path[:-4] + '_sep.csv'
    GT_path = path + '/test_labels.csv'
    model_gt = open(GT_path)
    model_outputs = open(outputs_path)
    outputs = []
    gt = []
    first_line = True
    output_csv = csv.reader(model_outputs)
    # skip_idxs = []
    skip_idx = []
    idx = 0
    for row in output_csv:
        if first_line:
            first_line = False
            continue
        try:
            outputs.append([float(item) for item in row])
        except:
            skip_idx.append(idx)
            print()
        idx += 1
    first_line = True
    gt_csv =  csv.reader(model_gt) 
    counter = 0
    idx
    for row in gt_csv:
        if first_line:
            first_line = False
            continue
        if counter in skip_idx:
            counter += 1
            continue
        gt.append([float(item) for item in row])
        counter += 1

    outputs = np.stack(outputs)
    gt = np.stack(gt)
    def rank_solver(solver, outputs, gt):
        total_time_gt = np.sum(gt, axis=0)
        total_time_pred = np.sum(outputs, axis=0)
        benchmarks = np.delete(total_time_gt, solver)
        gt_argsort = np.argsort(benchmarks)
        for i in range(len(benchmarks)):
            if total_time_pred[solver] < benchmarks[gt_argsort[i]]:
                break
            if i == 5:
                i += 1
        
        return i

    def evaluate(outputs, gt):
        pred_rank = []
        for i in range(7):
            pred_rank.append(rank_solver(i, outputs, gt))


        total_time_gt = np.sum(gt, axis=0, dtype=float)
        total_time_pred = np.sum(outputs, axis=0, dtype=float)
        # pred_ranked_solvers = np.argsort(total_time_pred)
        true_rankings = np.argsort(total_time_gt)
        true_rank = np.argsort(np.argsort(total_time_gt))

        print('true rank:', list(true_rank))
        print('pred_rank', pred_rank)

        print('true totals', total_time_gt)
        print('pred totals', total_time_pred)
        
        abs_diff = [np.abs(total_time_gt[i] - total_time_pred[i]) for i in range(len(total_time_gt))]
        import copy
        sep_mae = np.array(abs_diff)/len(outputs)
        abs_diff = np.sum(abs_diff)/len(outputs)
        print('absolute difference:', float(abs_diff))
        rank_disparity = 0
        for i in range(len(pred_rank)):
            rank_disparity += np.abs(pred_rank[i] - true_rank[i])
        
        print('rank disparity:', rank_disparity)
        
        avg_kt = 0
        pred_rankings = []
        # print(true_rankings)
        for i in range(len(pred_rank)):
            tmp = true_rankings
        
            tmp = true_rankings
            for j in true_rankings:
                if true_rankings[j] == i:
                    break
            tmp = np.delete(tmp, j)
            tmp = np.insert(tmp, pred_rank[i], i)
            pred_rankings.append(tmp)
        kts = []
        for r in pred_rankings:
            kt = stats.kendalltau(r, true_rankings)[0]
            kts.append(kt)
            avg_kt += kt
        avg_kt = avg_kt/len(pred_rankings)
        print('average kt:', avg_kt)
        print('all kts:', kts)

        data = {'abs. diff runtimes': [abs_diff], 'avg kts': [avg_kt], 'rank disparity': [rank_disparity/7]}
        # if os.path.isfile(results_path):
        df = pd.DataFrame(data)
        df.to_csv(results_path, mode = 'a', index=False, header=False)
        solvers = ['Solver 1', 'Solver 2', 'Solver 3', 'Solver 4', 'Solver 5', 'Solver 6', 'Solver 7']
        tuples = [(key, [value]) for i, (key, value) in enumerate(zip(solvers, sep_mae))]
        # breakpoint()
        # sep_df = pd.DataFrame(sep_mae, ['Solver 1', 'Solver 2', 'Solver 3', 'Solver 4', 'Solver 5', 'Solver 6', 'Solver 7'])
        sep_df = pd.DataFrame(dict(tuples))
        sep_df.to_csv(sep_results_path, mode = 'a', index=False, header=False)
        # for i in range(7):
        #     print('solver ' + str(i) + ' --')

    evaluate(outputs, gt)

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('datasize', type=int)
args=parser.parse_args()
print(args.datasize)
main(args.datasize)
    