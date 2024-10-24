#!/bin/bash
# param
maxNumJobs=200

mkdir -p log

class_name=${1}
num_iter=${2}
goal_time=${3}
dir_path=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/formulas/${class_name}
# core_dir_path=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/dataset/${class_name}_core
core_dir_path=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/dataset/${class_name}_core

for input in $dir_path/*.cnf; do
  cnf_name=${input##*/}
  cnf_name=${cnf_name%.cnf}
  # bash remove_core.sh $input ${core_dir_path}/${cnf_name::-3}.cnf $num_iter $goal_time &> log/${cnf_name}_rm.log &
  echo $input ${core_dir_path}/${cnf_name}.cnf $num_iter $goal_time $class_name $cnf_name
  break
  bash remove_core.sh $input ${core_dir_path}/${cnf_name}.cnf $num_iter $goal_time $class_name&> log/${class_name}_${cnf_name}_rm.log &
  runningJobs=`jobs -r | wc -l`
  echo "Running Jobs: ${runningJobs}"

  while [ `jobs -r | wc -l` -ge ${maxNumJobs} ]; do
    :
  done
done


while [ `jobs -r | wc -l` -gt 0 ]; do
  :
done

echo "Finish all jobs:"
jobs
echo "========================"
