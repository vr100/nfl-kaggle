#!/bin/bash
declare -a config=("gmm-after-pass-cfg.json" "gmm-after-snap-cfg.json" "gmm-before-pass-cfg.json" "gmm-before-snap-cfg.json" "gmm-between-snap-pass-cfg.json" "gmm-full-cfg.json")
declare -a output=("after_pass" "after_snap" "before_pass" "before_snap" "between_snap_pass" "full")

arraylength=${#config[@]}

for (( i=1; i<${arraylength}+1; i++ ));
do
  mkdir $output_path/${output[$i-1]}/
  cmd="python3 get-cluster.py --data_path $data_path --output_path $output_path/${output[$i-1]}/ --config_path $model_path/${output[$i-1]}/config.json --gmm_path $model_path/${output[$i-1]}/gmm.joblib"
  echo $cmd
  $cmd
done