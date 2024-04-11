#!/bin/bash


class_name=${1}
num_iter=${2}
goal_time=${3}
dir_path=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/formulas/${class_name}
# core_dir_path=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/dataset/${class_name}_core
core_dir_path=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/dataset/${class_name}_core

for input in $dir_path/*.cnf; do
  cnf_name=${input##*/}
  cnf_name=${cnf_name%.cnf}
  cnf_name=${cnf_name%_repeat*}
  echo $input 
  echo $cnf_name
  bash remove_core.sh $input ${core_dir_path}/${cnf_name}.cnf $num_iter $goal_time $class_name
  
done