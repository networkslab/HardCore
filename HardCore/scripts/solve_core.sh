#!/bin/bash
maxNumJobs=150
dir_path=${1}
core_path=${dir_path}_core
cd ..

for cnf in $dir_path/*.cnf; do
  echo $cnf
  cd "/home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/postprocess/cadical/build"
  #./cadical ../../../${cnf} --no-binary > '../../../'$dir_path'/solve.log'
  process_file(){
    cnf="$1"
    dir_path="$2"
    timeout 2000 ./cadical $cnf --no-binary ${cnf%.cnf}.drat &> $dir_path/solve.log
    echo $dir_path.log
    cd /home/joseph-c/sat_gen/CoreDetection/HardPSGEN/src/postprocess/drat-trim
    timeout 2000 ./drat-trim ${cnf} ${cnf%.cnf}.drat -c ${cnf%.cnf}_core
    rm -rf ${cnf%.cnf}.drat
    cd ../..
  }
  process_file "$cnf" "$dir_path" &
  runningJobs=`jobs -r | wc -l`
  echo "Running Jobs: ${runningJobs}"

  while [ `jobs -r | wc -l` -ge ${maxNumJobs} ]; do
    :
  done
done
wait

echo "Finish all jobs:"
jobs
echo "========================"