#!/bin/bash

no=0
no1=1

cnf_file=${1}
origin_core=${2}
num_iter=${3}
goal_time=${4}
class_name=${5}
goal_time=`echo $goal_time|awk '{print $1}'`

cnf_name=${cnf_file##*/}
cnf_name=${cnf_name%.cnf}
save_path=${cnf_file%/*}_post
mkdir $save_path

drat_file=${cnf_file%.cnf}.drat
core_file=${cnf_file%.cnf}_core
save_file=$save_path/${cnf_name}_r$no1.cnf
home=/home/joseph-c/sat_gen/CoreDetection/HardPSGEN

cd $home/src/postprocess/cadical/build
timeout 2500 ./cadical $cnf_file --no-binary $drat_file
cd $home/src/postprocess/drat-trim
timeout 2500 ./drat-trim $cnf_file $drat_file -c $core_file
cd $home/src/
python postprocess/sat_dataprocess.py --cnf $cnf_file --core $core_file --save $save_file --origin $origin_core --add_var
rm $drat_file
rm $core_file
cd ..

while [ "$no" -lt $num_iter ]; do
  no=$((no + 1))
  no1=$((no1 + 1))
  
  cnf_file=$save_file
  drat_file=${cnf_file%.cnf}.drat
  core_file=${cnf_file%.cnf}_core
  save_file=$save_path/${cnf_name}_r$no1.cnf

  cd $home/src/postprocess/cadical/build
  timeout 3000 ./cadical $cnf_file --no-binary $drat_file &>> $home/src/log/${class_name}_${cnf_name%.cnf}.log
  time=`tail -n 7 $home/src/log/${class_name}_${cnf_name%.cnf}.log`
  time=${time#*:}
  time=${time%%seconds*}
  time=`eval echo $time|awk '{print $1}'`
  
  if [ `echo "$time > $goal_time" | bc` -eq 1 ]
  then 
    rm $drat_file
    echo breaking for time $goal_time 
    break 1
  fi

  cd $home/src/postprocess/drat-trim
  timeout 3000 ./drat-trim $cnf_file $drat_file -c $core_file
  cd $home/src
  python postprocess/sat_dataprocess.py --cnf $cnf_file --core $core_file --save $save_file --origin $origin_core
  
  #if savefile is file
 
  if  [ ! -f $savefile ]; then
   
    # rm $core_file
    rm $drat_file
    cd ..
    echo breaking for $save_file
    break 1
    
  
  fi
  
  # rm $cnf_file
  # rm $core_file
  rm $drat_file  
  cd ..
    

done
