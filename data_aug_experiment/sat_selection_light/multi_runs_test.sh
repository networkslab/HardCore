#!/bin/bash

# exp_name=$1
# seed=604
# for i in 0 1 2 3 4 
# do
#     gpu=$((3+i))
#     version=seed_${seed}_split_${i}

#     echo "Testing..."
#     log_file=train_logs/${exp_name}_${version}_train.txt
#     python main.py --run_test -c configs/${exp_name}.yaml --seed $seed --version $version --device $gpu --split_idx $i &
# done

# exp_name=0731_hetero_lpe_SGD_1e-3_scheduler
exp_name=0731_hetero_lpe_SGD
seed=604

if [ ! -d lightning_logs/${exp_name} ]; then
  mkdir -p lightning_logs/${exp_name};
fi

echo "Testing..."
# python main.py -c configs/${exp_name}.yaml --seed $seed --version seed_${seed}_split_0 --device 2 --split_idx 0 --debug > lightning_logs/${exp_name}/split_0_log.txt &
# python main.py -c configs/${exp_name}.yaml --seed $seed --version seed_${seed}_split_1 --device 1 --split_idx 1 --debug > lightning_logs/${exp_name}/split_1_log.txt &
python main.py --run_test -c configs/${exp_name}.yaml --seed $seed --version seed_${seed}_split_2 --device 3 --split_idx 2 --debug &
# python main.py -c configs/${exp_name}.yaml --seed $seed --version seed_${seed}_split_3 --device 3 --split_idx 3 --debug > lightning_logs/${exp_name}/split_3_log.txt &
# python main.py -c configs/${exp_name}.yaml --seed $seed --version seed_${seed}_split_4 --device 4 --split_idx 4 --debug > lightning_logs/${exp_name}/split_4_log.txt &