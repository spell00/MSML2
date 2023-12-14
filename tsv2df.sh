#!/bin/bash
spd=200
experiment=old_data
split=1
fselection=mutual_info_classif
for ms in 1 2
do
  for mz in 10 1 0.1 0.01
  do
    for rt in 10 1 0.1
    do
      for mzp in 10 1 0.1 0.01
      do
        for rtp in 10 1 0.1
        do
          echo "mz${mz} rt${rt} mzp${mzp} rtp${rtp} ${spd}spd ms${ms} ${experiment}"
          if ! [[ -f "../../resources/$experiment/matrices/mz${mz}/rt${rt}/mzp${mzp}/rtp${rtp}/${spd}spd/ms${ms}" ]] &&
              (( $(echo "$rt >= $mz" |bc -l) )) && (( $(echo "$mzp >= $mz" |bc -l) )) && (( $(echo "$rtp >= $rt" |bc -l) )); then
            bash msml/preprocess/tsv2df.sh $mz $rt $mzp $rtp $spd $ms $experiment $split $fselection eco,sag,efa,kpn,blk,pool 0
          else
            if [[ -f "../../resources/$experiment/matrices/$mz/$rt/$mzp/$rtp/$spd/$ms" ]] ; then
              echo "Already exists"
            elif (( $(echo "$rt < $mz" |bc -l) )) ; then
              echo "rt < mz"
            elif (( $(echo "$mzp < $mz" |bc -l) )) ; then
              echo "mzp < mz"
            elif (( $(echo "$rtp < $rt" |bc -l) )) ; then
              echo "rtp < rt"
            else
              echo "whaaaat"
            fi
          fi
        done
      done
    done
  done
done
