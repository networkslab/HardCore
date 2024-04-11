import os

import csv
import numpy as np
import argparse 



def main(datasize):
    # gen_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/2k_bc_post_hardsatgen_csvs/runtimes_gen.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/2k_bc_post_hardsatgen_csvs/runtimes_og.csv'
    # # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/tseitin_formulas_nt_csvs/runtimes.csv'
    # combined_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/2k_bc_post_hardsatgen_csvs/runtimes_fixed.csv'
    # feat_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/satzilla_feats.csv'

    # gen_csv = '/home/joseph-c/sat_gen/CoreDetection/2k_bigcore_gen/runtimes.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/2k_bc_post_hardsatgen_csvs/runtimes_og.csv'
    # # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/tseitin_formulas_nt_csvs/runtimes.csv'
    # combined_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/runtimes.csv'
    # feat_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/satzilla_feats.csv'

    # hsgen_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/2k_bc_post_hardsatgen_csvs/runtimes_gen.csv'

    # gen_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_gen.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_og.csv'
    # # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/tseitin_formulas_nt_csvs/runtimes.csv'
    # combined_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_fixed.csv'
    # feat_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/satzilla_feats.csv'

    # gen_csv = '//home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/runtimes_gen.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/runtimes_og_v9.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/tseitin_formulas_nt_csvs/runtimes.csv'
    # combined_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_fixed.csv'
    # feat_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/satzilla_feats_v9.csv'


    gen_csv = '//home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/runtimes_gen_v2.csv'
    og_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/runtimes_og_v9.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/tseitin_formulas_nt_csvs/runtimes.csv'
    # combined_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_fixed.csv'
    feat_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/satzilla_feats_v9.csv'

    # gen_csv = '//home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/runtimes_gen_v2.csv'
    # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/runtimes_og_v4.csv'
    # # og_csv =  '/home/joseph-c/sat_gen/CoreDetection/tseitin_formulas_nt_csvs/runtimes.csv'
    # # combined_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_fixed.csv'
    # feat_csv =  '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/satzilla_feats_v4.csv'

    bc_csv = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/runtimes_gen_v2.csv'
    feat = open(feat_csv)
    lines = feat.readlines()
    skip_names = []
    all_feat_names = []
    og_files = []
    bc_files = []
    bc_idxs = []
    with open(bc_csv) as bc:
        reader = csv.reader(bc)

        counter = 0
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            name = row[1].split('_post')[0] + '.cnf'
            if name not in bc_files:
                bc_files.append(name)

    print('len bc files', len(bc_files))

    with open(og_csv) as og:
        reader = csv.reader(og)
        
        counter = 0
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            
            name = row[1]
            og_files.append(name)
    skip_first = True
    for line in lines:
        if skip_first:
            skip_first = False
            continue
        all_feat_names.append(line.split(',')[0])
        if len(line.split(',,')) > 1:
            skip_names.append(line.split(',')[0])
        if len(line.split(',-0.0,')) > 1:
            skip_names.append((line.split(',')[0]))
    feat.close()
    print('len og_files', len(og_files))
    print(skip_names)
    # all_rt_names = []
    # combined_rt = open(combined_csv)
    # lines = combined_rt.readlines()
    # skip_first=True
    # for line in lines:
    #     if skip_first:
    #         skip_first = False
    #         continue
    #     all_rt_names.append(line.split(',')[1])
    # if len(all_rt_names) != len(all_feat_names):
    #     print('uh oh length')
    # for i in range(len(all_rt_names)):
    #     if all_rt_names[i] != all_feat_names[i]:
    #         print('uh oh name', i, all_rt_names[i], all_feat_names[i])
    # ccsv = open(combined_csv, 'a')
    # combined_rt.close()
    # ccsv.write(',name,#var,#clause,base,HyWalk,MOSS,mabgb,ESA,bulky,UCB,MIN\n')
    # gen = open(gen_csv)
    # og = open(og_csv)
    # gen_reader = csv.reader(gen)
    # og_reader = csv.reader(og)
    gen_files = []
    num_gen = 0
    num_his = 0
    with open(gen_csv) as gen:
        reader = csv.reader(gen)    
        
        counter = 0
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            num_gen += 1
            try:
                # name = row[1].split('_sample')[0] + '.cnf'
                # name = row[1].split('_repeat')[0] + '.cnf' 
                name = row[1].split('_post')[0] + '.cnf'
            except:
                print()
            if name not in skip_names and name in og_files:
                gen_files.append(name)
            else:
                print('hi')
                num_his += 1
            writestr = ''
            for item in row:
                writestr += str(item) + ','
            writestr = writestr[:-1]
            writestr += '\n'
            # ccsv.write(writestr)
            
    # for file in gen_files:
    #     if file == gen_files[0]:
    #         print(file, skip_names[0])
    #     for skipfile in skip_names:
    #         if file == skipfile:
    #             print('uh oh')

    # print(len(gen_files))
    # num_gen = 25 # for hardsatgen
    print('num_gen', num_gen)
    print(len(skip_names))
    print('num his:', num_his)
    idx = num_gen 
    # idx = 50 ## for hardsatgen
    idxs = []
    prev = num_gen-1
    og_files = []
    skip_idxs = []
    with open(og_csv) as og:
        reader = csv.reader(og)
        
        counter = 0
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            
            name = row[1]
            if name in gen_files:                      
                idxs.append(idx)
            if name in bc_files:
                bc_idxs.append(idx)
            if name in skip_names:
                skip_idxs.append(idx)
            idx += 1
            writestr = ''
            row[0] = prev + 1
            for item in row:
                writestr += str(item) + ','
            writestr = writestr[:-1]
            writestr += '\n'
            # ccsv.write(writestr)
            prev += 1
    import random

    print('len bc idxs', len(bc_idxs))


            
    train_idxs = []
    val_idxs = []
    test_idxs = []
    # num_gen = 50 
    # num_og = 1619
    # total = 173381 # w2sat tset v9
    total = 16042 # combined v9
    # total = 10006 # hardsatgen v9
    # total = 167817# w2sat tseitin v8
    # total = 4442 # hardsatgen v8
    # total = 11122 # combined v8
    # total = num_gen + num_og + 16158
    # total = 78780 
    # total = 181178 #w2sat tseitin
    # total = 17030 #hardcore
    # total = 10350 #hardsatgen tseitin
    # total = 173725 # w2sat v4
    # total = 21664 # combied tseitin v6
    # total = 18510 # combined tseitin v7
    # total = 175205 # w2sat v7
    # total = 14984 # hardsatgen v6
    # total = 178359 # w2sat v6
    # total = 186030
    # lec_1k_idxs = range(num_gen + num_og, total )
    lec_1k_idxs = range(num_gen, total )
    og_idxs = idxs.copy()
    print('length of idxs: ', len(idxs))

    print(len(gen_files))
    print(len(og_idxs))
    test_opts = set(lec_1k_idxs).difference(set(og_idxs))
    test_opts = test_opts.difference(set(skip_idxs))
    test_opts = list(test_opts)

    # gen_files = gen_files[:int(len(gen_files)*0.21)]
    for i in range(50):
        # --- for HARDSATGEN ---- 

        # og_idxs = idxs + random.sample(bc_idxs, k=datasize-10)

        # print('length og_idxs:', len(og_idxs))
        # print('len genfiles',len(gen_files))

        # test_opts = set(lec_1k_idxs).difference(set(og_idxs))
        # test_opts = list(test_opts)
        # ----- end for HARDSATGEN ----

        used_gen_files = []
        gen_idx = []
        used_idxs = []
        idx = num_gen 
        # idx = 50 #for hardsatgen 
        rand_idx = list(range(len(gen_files)))
        random.shuffle(rand_idx)
        for i in rand_idx:
            if gen_files[i] not in used_gen_files:
                used_gen_files.append(gen_files[i])
            # if len(used_gen_files) > (datasize + 10):
            if len(used_gen_files) > datasize:
            # if len(used_gen_files) > 0:
                break
        print('len used gen files', len(used_gen_files))
        with open(og_csv) as og:
            reader = csv.reader(og)
            
            counter = 0
            for row in reader:
                if counter == 0:
                    counter += 1
                    continue
                
                name = row[1]
                if name in used_gen_files:
                    used_idxs.append(idx)
                
                idx += 1
                
                    
                writestr = ''
                row[0] = prev + 1
                for item in row:
                    writestr += str(item) + ','
                writestr = writestr[:-1]
                writestr += '\n'
                # ccsv.write(writestr)
                prev += 1

        train_gen= []
        genfile_num = {}
        for i in range(len(gen_files)):
            if gen_files[i] in used_gen_files:
                if gen_files[i] not in genfile_num.keys():
                    genfile_num[gen_files[i]] = 1
                else:
                    genfile_num[gen_files[i]] += 1
                if genfile_num[gen_files[i]] < 4:
                    train_gen.append(i)
            # train_idx = list(range(len(gen_files)))
    
        random.shuffle(og_idxs)
        # train_idx = list(range(len(gen_files))) + og_idxs[:int(0.8*len(og_idxs))]
        

        #---- for hardsatgen
        # random.shuffle(train_gen)
        # train_gen = train_gen[:int(datasize*4/1000)]
        # # train_idx = train_gen + used_idxs[:int(0.8*len(used_idxs))] + og_idxs[:int(0.8*len(og_idxs))]
        # train_idx = train_gen + og_idxs[:int(0.8*len(og_idxs))] + used_idxs
        # train_idx = (train_gen + og_idxs[:int(0.8*len(og_idxs))]) #--this one

        # train_idx = og_idxs[:int(0.8*len(og_idxs))]
        

        #----- end for hardsatgen
        random.shuffle(used_idxs)
        random.shuffle(train_gen)
        train_idx = train_gen + used_idxs[:int(0.8*len(used_idxs))]

        # train_idx = train_gen 

        # train_idx = used_idxs[:int(0.8*len(used_idxs))] + random.sample(list(range(25)), k=np.max([1,int(datasize*2/1000)])) #for strict hardsatgen
        # train_idx = used_idxs[:int(0.8*len(used_idxs))] + list(range(25)) #for 50 hardsatgen
        # 

        # 
        print('used_idxs length', len(used_idxs))
        print('train_gen length', len(train_gen))
    

        # print(len(used_idxs))
        # train_idx = used_idxs[:int(0.8*len(used_idxs))] 

        # train_idx = random.sample(list(range(50)), k=5)
        # train_idx = list(range(25))
        # train_idx = train_gen
        # print(int(0.05*len(og_idxs)))
        # train_idx = og_idxs[:50]
        print('length of train set: ', len(train_idx))
        random.shuffle(train_idx)
        val_idx = used_idxs[int(0.8*len(used_idxs)):]

        # val_idx = used_idxs[:40]

        # val_idx = used_idxs

        # --- for hardsatgen:
        # val_idx = used_idxs[int(0.8*len(used_idxs)):] + og_idxs[int(0.8*len(og_idxs)):]
        # val_idx = (og_idxs[int(0.8*len(og_idxs)):]) # --this one
        # --- end for hardsatgen
        random.shuffle(test_opts)
        test_idx = test_opts[:10000]
        # test_idx = og_idxs[int(0.8*len(og_idxs)):]
        train_idxs.append(train_idx)

        print('val set len', len(val_idx))
        val_idxs.append(val_idx)
        test_idxs.append(test_idx)
        print('test length', len(test_idxs[-1]))
    # data_dir  = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/'
    # data_dir  = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_2k_bc/'
    # data_dir  = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/'
    # data_dir  = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og/'
    # data_dir = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/'
    # data_dir = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_og/'
    data_dir = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/'
    # data_dir = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/'
    train_idx = np.stack(train_idxs)
    val_idx = np.stack(val_idxs)
    test_idx =np.stack(test_idxs)
    train_idx.dump(data_dir + '/split_idx/train_idx.npy')
    val_idx.dump(data_dir + '/split_idx/val_idx.npy')
    print(data_dir)
    # test_idx = np.load('/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og/split_idx/test_idx.npy', allow_pickle=True)
    test_idx.dump(data_dir + '/split_idx/test_idx.npy')

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('datasize', type=int)
args=parser.parse_args()
# print(args.datasize)
main(args.datasize)