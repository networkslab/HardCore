#!/bin/bash
# 100 200 300
for j in  10 20 30 40 50 
do

# j=401
    python /home/joseph-c/sat_gen/CoreDetection/public_repo/satzilla_combined_data.py $j
    for i in {0..14}
    do
        python /home/joseph-c/sat_gen/CoreDetection/public_repo/sat_selection_light/main.py --split_idx $i --device 6
        python /home/joseph-c/sat_gen/CoreDetection/public_repo/satzilla_bench_ranking.py $j
    done
done