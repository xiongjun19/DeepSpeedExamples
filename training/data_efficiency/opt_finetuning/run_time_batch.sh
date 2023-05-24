set -x

# sh bash_script/run_opt_time.sh opt-1.3b config/ds_config_opt_1.3b_cpu.json 1 > log_cpu1.txt
sh bash_script/run_opt_time.sh opt-1.3b config/ds_config_opt_1.3b_cpu_para2.json 2  > log_cpu2.txt
sh bash_script/run_opt_time.sh opt-1.3b config/ds_config_opt_1.3b_cpu_para8.json 8 > log_cpu8.txt
# sh bash_script/run_opt_time.sh opt-1.3b config/ds_config_opt_1.3b_disk.json 1 > log_disk_1.txt 
sh bash_script/run_opt_time.sh opt-1.3b config/ds_config_opt_1.3b_disk_para2.json 2  > log_disk2.txt
sh bash_script/run_opt_time.sh opt-1.3b config/ds_config_opt_1.3b_disk_para8.json 8 > log_disk8.txt
