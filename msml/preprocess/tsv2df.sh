#!/bin/bash
START_TIME=$SECONDS

mz_bin="$1"
rt_bin="$2"
mz_bin_post="$3"
rt_bin_post="$4"
spd="$5"
ms_level="$6"
experiment="$7"
split_data="$8"
feature_selection="$9"
feature_selection_threshold="${10}"
run_name="${11}"
test="${12}"
path="${13}"

if [ "$spd" == "" ]; then
  spd=200
fi
if [ "$ms_level" == "" ]; then
  ms_level=2
fi
if [ "$experiment" == "" ]; then
  experiment="old_data"
fi
if [ "$split_data" == "" ]; then
  split_data=0
fi
if [ "$feature_selection" == "" ]; then
  feature_selection="mutual_info_classif"
fi
if [ "$feature_selection_threshold" == "" ]; then
  feature_selection_threshold="0.9"
fi
if [ "$run_name" == "" ]; then
  run_name="eco,sag,efa,kpn,blk,pool"
fi
if [ "$test" == "" ]; then
  test=0
fi

# When split_data is 1, then the data has already been presplit into train/valid/test
if [ "$split_data" == 1 ]; then
  echo "split_data: $split_data"
  if [ "$ms_level" == 1 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection_threshold=$feature_selection_threshold --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3 msml/preprocess/make_tensors_ms1_split.py --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --feature_selection_threshold=$feature_selection_threshold --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin --resources_path=$path
  fi
  if [ "$ms_level" == 2 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection_threshold=$feature_selection_threshold --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3 msml/preprocess/make_tensors_ms2_split.py --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --feature_selection_threshold=$feature_selection_threshold --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin --resources_path=$path
  fi
fi
if [ "$split_data" == 0 ]; then
  echo "split_data: $split_data"
  if [ "$ms_level" == 1 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection_threshold=$feature_selection_threshold --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3 msml/preprocess/make_tensors_ms1.py --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --feature_selection_threshold=$feature_selection_threshold --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin --resources_path=$path
  fi
  if [ "$ms_level" == 2 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection_threshold=$feature_selection_threshold --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3 msml/preprocess/make_tensors_ms2.py --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --feature_selection_threshold=$feature_selection_threshold --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin --resources_path=$path
  fi
fi

