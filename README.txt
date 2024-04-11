Work-In-Progress: streamlining the pipeline for user convenience. 

full anaconda environment provided in enviromnent.yml

cadical: clone to ./postprocess/cadical
	build: ./configure && make
	make dir log under ./postprocess/cadical/
drat-trim: clone to ./postprocess/drat-trim
	build: make

generating the Tseitin data:
	- /HardCore/tseitin/cnfgen_tseitin.py:
		- edit path at l. 34, 48, 49, 53
		- run python cnfgen_tseitin.py  //this script generates the tseitin cnfs
	- /HardCore/tseitin/run_original.py:
		- edit path at 35, 37, 38, 40
		- run python run_original.py //this script runs the 7 solvers and saves their logs
	- /HardCore/tseitin/parse.py
		- edit path at l. 8, 10, 11
		- run python parse.py // this script parses solver logs and collects runtime information
	- /HardCore/tseitin/fix_csvs.py 
		- edit path at l. 5, 6
		- run fix_csvs.py // this script re-formats the csv output by parse.py for use 
	- /HardCore/tseitin/filter_triv_to.py:
		- edit path at l. 5,6,8
		- run python filter_triv_to //this script filters the tseitin cnfs which have excessively low runtimes
preparing the data:
Place
	- /HardCore/scripts/solve_core.sh
		- bash solve_core.sh dataset/{data_name} // this script retrieves the cores
Training the model:
retrieving training data:
	- /HardCore/run_add_core.py: 
		- place cnf dataset at /HardCore/src/dataset/
		- edit source_dir at l. 217
		- edit path at l. 248
		- run run_add_core.py // this script adds generated data to cores
	- /HardCore/scripts/auto_rmcore_multi_train_gen.sh
		- set maxNumJobs at l. 3 to desired parallelization
		- run bash auto_rmcore_multi_train_gen.sh data_name 200 1000 //this script does Core Refinement with classical Core Detection and saves the pairs
	- /HardCore/scripts/process_data.py
		- edit path at l. 432, 434, 437, 447, 151, 152, 153, 50
		- edit l. 500 to desried parallelization
		- run python process_data.py
	- /training/post_model_run.py
		- edit paths at l. 10, 11, 92, 223
		- run post_model_run.py // this script trains the GNN model
generating new data: 
	- /HardCore/run_add_core.py: 
		- place cnf dataset at /HardCore/src/dataset/
		- edit source_dir at l. 217
		- edit path at l. 248
		- run run_add_core.py
	- /HardCore/remove_neurocore.py:
		- edit path at l. 58, 528, 546, 568, 208
		- edit saved model's path at l. 444
		- run python remove_neurocore.py

Data Augmentation Experiment:

preparing the data: 
runtime data must be placed in a csv with the following columns:
,name,#var,#clause,base,HyWalk,MOSS,mabgb,ESA,bulky,UCB,MIN
	-/data_aug_experiment/satzilla_feats.py:
		- edit paths at l. 179, 180, 181
	- run python data_aug_experiment

running the experiment:
	- /data_aug_experiment/satzilla_combined_data.py:
		-edit paths at l. 359, l. 37-41
		-edit naming convention string at l. 131. For HardSATGEN the convention is ''_post'', w2_sat uses ''_sample'', HardSATGEN uses ''_repeat''
		-l. 201: edit the total number of cnfs in the data
		-for HardSATGEN:
			-uncomment l. 154
			-uncomment l. 315 (strict) or 316 (loose)
		-for testing without augmentation:
			-uncomment l. 325
	-/data_aug_experiment/sat_selection_light/dataset/satzilla.py
		-edit paths at l. 19-23
	-/data_aug_experiment/satzilla_bench_ranking.py:
		-edit path at l. 25
	-/data_auga_experiment/run_bench_exp.bash:
		-edit paths
		-run bash run_bench_exp.bash 
plotting results:
	-/data_aug_experiment/plot_performance.py:
		-edit paths lines 7-12, 41
	
	


	