set -x

sh bash_script/run_opt_prof.sh opt-1.3b config/ds_config_opt_1.3b_cpu.json 1 'cpu' 
# sh bash_script/run_opt_prof.sh opt-1.3b config/ds_config_opt_1.3b_cpu_para2.json 2 'cpu'
# sh bash_script/run_opt_prof.sh opt-1.3b config/ds_config_opt_1.3b_cpu_para8.json 8 'cpu'
# sh bash_script/run_opt_prof.sh opt-1.3b config/ds_config_opt_1.3b_disk.json 1 'disk'
# sh bash_script/run_opt_prof.sh opt-1.3b config/ds_config_opt_1.3b_disk_para2.json 2 'disk'
# sh bash_script/run_opt_prof.sh opt-1.3b config/ds_config_opt_1.3b_disk_para8.json 8 'disk'
