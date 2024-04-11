import pickle
import torch
from model import SATInstanceEncoderHetero
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os


data_dir = '/home/joseph-c/sat_gen/CoreDetection/training/tseitin_training_graphs/'
og_dir = '/home/joseph-c/sat_gen/CoreDetection/training/tseitin_train_og/'
train_files = []
test_files = []
all_files = []
import random
for filename in os.listdir(og_dir):
    if filename[-3:] == 'cnf':
        all_files.append(filename[:-4])
random.shuffle(all_files)
train_files = all_files[:int(len(all_files)*0.8)]
test_files = all_files[int(len(all_files)*0.8):]
# train_files = ['mp1-squ_any_s09x07_c27_abix_UNS', 'mp1-tri_ali_s11_c35_abix_UNS']
# test_files = ['mp1-tri_ali_s11_c35_bail_UNS']


# train_files = all_files
# test_files = []




# #next: dataload
# dataset = [Data(hetero_graphs[i], y=node_labels[i]) for i in range(len(hetero_graphs))]
# trainloader = DataLoader(dataset)
# graph = next(iter(trainloader))
# print()

def evaluate(model, graph, features, labels, thresh, verbose=False, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        
        pred = torch.where(logits > thresh, 1, 0)
        # n_TP = torch.sum(pred==labels * labels == 1)
       
        correct = torch.sum(pred == labels)
        acc = correct.item() * 1.0 / len(labels)
        n_pred_core = int(torch.sum(pred))
        n_true_core = int(torch.sum(labels))
        n_TP = 0
        for i in range(len(pred)):
            if pred[i] == labels[i] and pred[i] == 1:
                n_TP += 1
        n_TP = n_TP/n_true_core
        if verbose:
            print('acc: ', acc)
            print("pred core size:", n_pred_core )
            print("true core size:", n_true_core)
            print("Core recovered:", np.round(n_TP, decimals=2))
            print('Pred Core ratio', np.round(n_pred_core/len(pred), decimals=2))
            print('Total number of clauses:', len(pred))
            print('Core ratio', np.round(n_true_core/len(pred), decimals=2))
            
        return n_pred_core, n_true_core, len(pred), np.round(n_TP, decimals=2), np.round(n_pred_core/len(pred), decimals=2), np.round(n_true_core/len(pred), decimals=2)
    

# print(num_vars, num_clause)
# for k in range(1):
#     for j in range(1):
device = torch.device('cuda:2')
model = SATInstanceEncoderHetero(32, 3, 'nada', device=device, act='mean').to(device)
# stack_model =SATInstanceEncoderHetero(32, 3, 'nada', device=device, act='mean').to(device)
h_in = []
# for i in range(len(hg_info)):
#     num_vars, num_clause = hg_info[i]
#     h_in.append({'pos_lit': torch.ones((num_vars, 3), device=device),
#         'neg_lit': torch.ones((num_vars, 3), device=device),
#         'clause': torch.ones((num_clause, 3), device=device)})

# print(h_in.keys())
from torch.optim import lr_scheduler

opt = torch.optim.Adam(model.parameters(), lr=0.003)
scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)
# all_idx = list(range(len(hetero_graphs)))
# random.shuffle(all_idx)
# train_idx = all_idx[:int(len(hetero_graphs)*0.8)]
# test_idx = all_idx[int(len(hetero_graphs)*0.8):]
import glob 
thresh = 0.5
epoch_counter = 0
data_dir = '/home/joseph-c/sat_gen/CoreDetection/training/tseitin_training_graphs/'
os.chdir(data_dir)
train_list = []
test_list = []
print('assembling trainset')
for file in tqdm(train_files):
    # print(file)
    # continue
    for filename in glob.glob(data_dir + '/' + 'hetero_graphs_tseitin_train_post_'+file+'*'): 
        train_list.append(filename)
random.shuffle(train_list)
for file in tqdm(test_files):
    for filename in glob.glob(data_dir + '/' + 'hetero_graphs_tseitin_train_post_' + file + '*'):
        test_list.append(filename)
print('training')
for epoch in range(1):
# if 1 < 0:
    for filename in tqdm(train_list):
        # print(file)
        # continue
        # for filename in glob.glob('hetero_graphs_'+file+'*'): 
        if 1 > 0:       
            model.train()
            # print(filename)
            try:
                hetero_graphs = pickle.load(open(os.path.join(data_dir, filename), 'rb'))
                hg_info = pickle.load(open(os.path.join(data_dir, 'hg_info_'+filename.split('hetero_graphs_')[1]), 'rb'))
            
                node_labels = pickle.load(open(os.path.join(data_dir, 'node_labels_'+filename.split('hetero_graphs_')[1]), 'rb'))
            except:
                print('failed to loaod data')
                continue
            num_vars, num_clause = hg_info
            h_in = {'pos_lit': torch.ones((num_vars, 3), device=device),
            'neg_lit': torch.ones((num_vars, 3), device=device),
            'clause': torch.ones((num_clause, 3), device=device)}
            # forward propagation by using all nodes
            logits = model(hetero_graphs.to(device), h_in)
            # core_size_reg = F.mse_loss(torch.mean(logits), torch.mean(node_labels[epoch]))
            # confidence_reg = -F.mse_loss(logits, torch.ones((len(logits),1),device=device)*0.5)
            confidence_reg = F.mse_loss(logits, torch.ones((len(logits),1),device=device))
            num_vars, num_clause = hg_info
            num_pred = torch.sum(logits)
            # print('num_clause', num_clause)
            big_reg = 0
            small_reg = 0
            # avg_reg = F.mse_loss(torch.mean(logits), torch.mean(node_labels[epoch]))
            one_reg = F.mse_loss(logits, torch.zeros(logits.shape, device=device))
            BCE = F.binary_cross_entropy(logits, node_labels.to(device))
            
            
            # compute loss
            # loss = 4*BCE  + 0.5*confidence_reg + 2*one_reg
            loss = BCE
            # loss = BCE
            # compute validation accuracy
            
            if epoch_counter%200 == 0:
                print(filename)
                print('loss:', loss)
                # print('conf_reg', confidence_reg)
                # print('big/small', big_reg, small_reg)
                # print('core_size_reg', core_size_reg)
                # print([int(logit) for logit in logits])
                # print('one reg', one_reg)
                # print(logits)
                evaluate(model, hetero_graphs.to(device), h_in, node_labels.to(device), thresh, verbose=True, device=device)
            else:
                evaluate(model, hetero_graphs.to(device), h_in, node_labels.to(device), thresh, device=device)
            # input()
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
            epoch_counter += 1
    scheduler.step()
results = []
print("testing ---------------")
epoch_counter = 0
model.eval()
for filename in tqdm(test_list):
    # print(file)
    # continue
    # for filename in glob.glob(data_dir + 'hetero_graphs_'+file+'*'):        
        # print(filename)
        try:
            hetero_graphs = pickle.load(open(os.path.join(data_dir, filename), 'rb'))
            hg_info = pickle.load(open(os.path.join(data_dir, 'hg_info_tseitin_train_post_'+filename.split('hetero_graphs_tseitin_train_post_')[1]), 'rb'))
        
            node_labels = pickle.load(open(os.path.join(data_dir, 'node_labels_tseitin_train_post_'+filename.split('hetero_graphs_tseitin_train_post_')[1]), 'rb'))
        except:
            
            print('failed to loaod data')
            continue
        num_vars, num_clause = hg_info
        h_in = {'pos_lit': torch.ones((num_vars, 3), device=device),
        'neg_lit': torch.ones((num_vars, 3), device=device),
        'clause': torch.ones((num_clause, 3), device=device)}
        # results.append(evaluate(model, hetero_graphs.to(device), h_in, node_labels.to(device), thresh, verbose=True, device=device))
        if epoch_counter%200 == 0:
            results.append(evaluate(model, hetero_graphs.to(device), h_in, node_labels.to(device), thresh, verbose=True, device=device))
        else:
            results.append(evaluate(model, hetero_graphs.to(device), h_in, node_labels.to(device), thresh, verbose=False, device=device))
        epoch_counter += 1
# print('test files:', test_files)

print("done --------------------")
results_np = np.array(results)
print('average test core recovered:', np.mean(results_np[:,3]))
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
# print('average')
# print("average test core coverage:", np.mean(n_TPs))
import pandas as pd
import time
result_csv = pd.DataFrame(results, columns=['pred core size', 'true core size', 'cnf size', 'core recovered', 'pred core ratio', 'true core ratio' ])
save_file = '/home/joseph-c/sat_gen/CoreDetection/training/results/' + data_dir.split('/')[-2]+str(time.ctime())  + '.csv'
result_csv.to_csv(save_file, header=True)

# torch.save(model, '/home/joseph-c/sat_gen/CoreDetection/training/model_'+ str(time.time()) + '_post_model_mp1')
# pickle.dump(train_files, open('/home/joseph-c/sat_gen/CoreDetection/training/train_files.pkl', 'wb'))
# pickle.dump(test_files, open('/home/joseph-c/sat_gen/CoreDetection/training/test_files.pkl', 'wb'))


'''
1. original instance -> original core -> original core + ps_gen 
2. model(original core + ps_gen) -> predicted core
3. un-core(predicted core \ original core)



'''