#!/bin/bash
spd=200
experiment=old_data
pass_exist=FALSE
split_data=1
for ms in 1 2
do
  for mz in 10 1 0.1 0.01
  do
    for rt in 10 1 0.1
    do
      path="../../resources/$experiment/tsv/mz${mz}/rt${rt}/${spd}spd/ms${ms}"
      if ! [[ -e $path ]] || [ $pass_exist == FALSE ]; then
        # We only do retention time equal or higher than the mz
        if (( $(echo "$rt >= $mz" |bc -l) )); then
          # echo $path
          echo "mz${mz} rt${rt} ${spd}spd ms${ms} ${experiment}"
          bash msml/preprocess/mzdb2tsv.sh $mz $rt $spd $ms $experiment $split_data
        else
          echo "$rt < $mz"
        fi
      else
        echo "Already exists: mz${mz} rt${rt} ${spd}spd ms${ms} ${experiment}"
      fi
    done
  done
done

