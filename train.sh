#!/bin/bash
spd=200
experiment=old_data
fselection=mutual_info_classif
# berm=combat
preprocess_scaler=none
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
          for tl in 0 1
          do
            for pred_test in 0 1
            do
              for dann_plate in 0 1
              do
                for zinb in 0 1
                do
                  for variational in 0 1
                  do
                    echo "mz${mz} rt${rt} mzp${mzp} rtp${rtp} ${spd}spd ms${ms} ${experiment}"
                    if ! [[ -f "../../resources/$experiment/matrices/mz${mz}/rt${rt}/mzp${mzp}/rtp${rtp}/${spd}spd/ms${ms}" ]] &&
                        (( $(echo "$rt >= $mz" |bc -l) )) && (( $(echo "$mzp >= $mz" |bc -l) )) && (( $(echo "$rtp >= $rt" |bc -l) )); then
                      python3.8 msml/dl/train/train_ae_classifier.py --mz_bin=$mz --rt_bin=$rt --preprocess_scaler=$preprocess_scaler --triplet_loss=$tl --predict_tests=$pred_test --dann_sets=0 --balanced_rec_loader=0 --dann_plates=$dann_plate --zinb=$zinb --variational=$variational --feature_selection=$fselection --use_valid=1 --use_test=1
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
        done
      done
    done
  done
done
