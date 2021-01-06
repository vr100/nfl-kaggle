#!/bin/bash
declare -a config=("gmm-after-pass-cfg.json" "gmm-after-snap-cfg.json" "gmm-before-pass-cfg.json" "gmm-before-snap-cfg.json" "gmm-between-snap-pass-cfg.json" "gmm-full-cfg.json")
declare -a output=("after_pass" "after_snap" "before_pass" "before_snap" "between_snap_pass" "full")

arraylength=${#config[@]}

for (( i=1; i<${arraylength}+1; i++ ));
do
  cmd="python3 run-gmm.py --data_path $data_path --output_path $output_path/${output[$i-1]}/ --config_path $config_path/${config[$i-1]}"
  echo $cmd
  $cmd
done