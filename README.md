
## Authors

* Simon Pelletier (2021-) (Current Maintainer)

# Prerequisites on ubuntu
`apt-get install -y parallel`<br/>
`apt-get install -y python3`<br/>
`apt-get install -y python3-pip`<br/>
`apt-get install -y r-base`<br/>
`apt-get purge -y openjdk-\*`<br/>
`apt install -y openjdk-8-jre`<br/>
`apt install -y openjdk-8-jdk`<br/>
`apt-get install -y ant`<br/>
`apt-get install -y ca-certificates-java`<br/>
`update-ca-certificates -f`<br/>
`chmod +x mzdb2train.sh`<br/>

`chmod +x msml/scripts/mzdb2tsv/amm`

# Install python dependencies
`pip install -r requirements.txt`


On Windows:
The first step needs to be executed on Windows because it calls raw2mzdb.exe and the software only exists for Windows.

In a  Windows PowerShell:


`./msml/preprocess/raw2mzdb.bat $spd $group $path`

Where $spd is samples per day, $experiment is the experiment name (e.g. old_data) and $path is the path to the raw files.

For example:
`./msml/preprocess/raw2mzdb.bat 200 old_data ..\resources`


The resulting mzdb files are stored in `$path/mzdb/$spd/$group/`

On Linux (tested with WLS Ubuntu 20.04):

`bash ./msml/preprocess/mzdb2tsv.sh $path $mz_bin $rt_bin $spd $ms_level $experiment $split_data`

Where
$path is the path to the raw files
$mz_bin is the bin size for the m/z axis
$rt_bin is the bin size for the retention time axis
$spd is samples per day
$ms_level is the ms level to extract
$experiment is the experiment name
$split_data is a boolean to split the data into train, valid and test sets. If 0, it is assumed that the data
is already split, i.e. you would have `$path/mzdb/$spd/$group/train/`, `$path/mzdb/$spd/$group/valid/` and 
`$path/mzdb/$spd/$group/test/` folders.

The resulting tsv files are stored in `$path/tsv/$spd/$group/all/` or `$path/tsv/$spd/$group/train/` or 
`$path/tsv/$spd/$group/valid/` or `$path/tsv/$spd/$group/test/` depending on the value of $split_data.

To make a matrix file containing only the most important features, without preserving the spatial organization of the 
data, run the following command:

`python3 msml/preprocess/tsv2df.sh $mz_bin $rt_bin $mz_bin_post $rt_bin_post $spd $ms_level $experiment $split_data $feature_selection $feature_selection_threshold $run_name $test $path`

Where
$mz_bin is the bin size for the m/z axis. This is the bin size that was used to create the tsv files. To save time and space if one wants to test multiple bin sizes, lower bin sizes to make the tsv file can be used to create matrix file that use higher bin sizes. For example, if the tsv files were created with a bin size of 0.01, one can use a bin sizes of 0.1 or 1 to create the final matrix file.
$rt_bin is the bin size for the retention time axis. This is the bin size that was used to create the tsv files. To save time and space if one wants to test multiple bin sizes, lower bin sizes to make the tsv file can be used to create matrix file that use higher bin sizes. For example, if the tsv files were created with a bin size of 0.01, one can use a bin sizes of 0.1 or 1 to create the final matrix file.
$mz_bin_post is the final bin size for the m/z axis
$rt_bin_post is the final bin size for the retention time axis
$spd is samples per day
$ms_level is the ms level to extract
$experiment is the experiment name
$split_data is a boolean to split the data into train, valid and test sets. If 0, it is assumed that the data is already split, i.e. you would have `$path/tsv/$spd/$group/train/`, `$path/tsv/$spd/$group/valid/` and `$path/tsv/$spd/$group/test/` folders.
$feature_selection is a string that indicates which feature selection algorithm to use. Must be either mutual_info_classif or f_classif.
$feature_selection_threshold is a float that indicates the threshold to use for the feature selection algorithm.
$run_name is the name of the run. It is used to name the matrix file.
$test is to indicate if the run is a test, which would use less data to go faster
$path is the path to the tsv files

Command line example:
`bash msml/preprocess/tsv2df.sh 1 1 1 1 200 2 old_data 0 mutual_info_classif 0.3 first_run 0 ..\..\..\resources`


## Train deep learning model
Command line example:

`python3 msml\dl\train\mlp\train_ae_classifier.py --triplet_loss=1 --predict_tests=1 --dann_sets=0 --balanced_rec_loader=0 --dann_batches=0 --zinb=0 --variational=0 --use_valid=1 --use_test=1`

For your data to work, it should be a matrix: rows are samples, columns are features. Feature names can be whatever,
but the row names (in the first column named ID), the names should be as such: `{experiment_name}_{class}_{batch_number}_{id}`

*** The batch number should start with the letter `p`, followed by batch number. This is because for the experiment
it was designed for, the batches were the plates in which the bacteria grew. It should change soon!
e.g.: `rd159_blk_p16_09`

## Observe results from a server on a local machine 
On local machine:<br/>
`ssh -L 16006:127.0.0.1:6006 simonp@192.168.3.33`

On server:<br/>
`python3 -m tensorboard.main --logdir=/path/to/log/file`

Open in browser:<br/>
`http://127.0.0.1:16006/`

![](E:\GITLAB\MSML\images\ae-dann.png "Autoencoder-DANN")

## Hyperparameters
    thres (float): Threshold for the minimum number of 0 tolerated for a single feature. 
                   0.0 <= thres < 1.0
    dropout (float): Number of neurons that are randomly dropped out. 
                     0.0 <= thres < 1.0
    smoothing (float): Label smoothing replaces one-hot encoded label vector 
                       y_hot with a mixture of y_hot and the uniform distribution:
                       y_ls = (1 - α) * y_hot + α / K
    margin (float): Margin for the triplet loss 
                    (https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905)
    gamma (float): Controls the importance given to the batches adversarial loss
    beta (float): Controls the importance given to the Kullback-Leibler loss
    zeta (float): Controls the importance given to the ZINB loss
    nu (float): Controls the importance given to the ZINB loss
    layer1 (int): The number of neurons the the first hidden layer of the encoder and the
                  last hidden layer of the decoder
    layer2 (int): The number of neurons the the second hidden layer (Bottleneck)
    ncols (int): Number of features to keep
    lr (float): Model's optimization learning rate
    wd (float): Weight decay value
    scale (categorical): Choose between ['none', 'binarize', 'robust', 'standard', 'l1']
    dann_sets (boolean): Use a DANN on set appartenance?
    dann_batches (boolean): USe a DANN on 
    zinb (boolean): Use a zinb autoencoder?
    variational (boolean): Use a variational autoencoder?
    tied_w (boolean): Use Autoencoders with tied weights?
    pseudo (boolean): Use pseudo-labels?
    tripletloss (boolean): Use triplet loss?
    train_after_warmup (boolean): Train the autoencoder after warmup?

## Metrics:
Rec Loss (Reconstruction loss)
Domain Loss: Should be random (e.g. ~0.6931 if 2 batches)
Domain Accuracy: Should be random (e.g. ~0.5 if 2 batches)
(Train, Valid or Test) Loss (l, h or v): Classification loss for low (l), high (h) or very high (v) concentrations
                                         of bacteria in the urine samples. These are the subcategories for the 
                                         data in example_resources, but it might be different if other subcategories 
                                         are different (subcategories are optional).
(Train, Valid or Test) Accuracy (l, h or v): Classification accuracies
(Train, Valid or Test) MCC (l, h or v): Matthews correlation coefficients


## Dockerfiles
Place the folder `resources` in the same folder as the `MSML` repository. <br>
Inside `resources`, it must contain: `resources/[exp_name]/mzdb/*.mzdb`, 
where `exp_name` is the name of the experiment. All raw files must have already 
been processed into `mzdb` files

From outside the repo: <br>
- Create a Volume names `resources` that will mount the data<br>
`docker volume create --opt type=none --opt o=bind --opt device=${pwd}/resources resources`<br>
- Build the docker image `mzdb2tsv`
`docker build MSML2 -f MSML2\Dockerfile.mzdb2tsv -t mzdb2tsv`<br>
- Run a container from the image. The volume created earlier must be mounted. <br>
`docker run --mount source=resources,destination=/resources mzdb2tsv` <br>
- Build the docker image `tsv2df`<br>
`docker build MSML2 -f MSML2\Dockerfile.tsv2df -t tsv2df`
- Run a container from the image. The volume created earlier must be mounted. <br>
`docker run --mount source=resources,destination=/resources tsv2df` <br>

TO REMOVE
`./msml/preprocess/raw2mzdb.bat 200 old_data 1 200 0.2 20 0.2 20 0 mutual_info_classif "eco,sag,efa,kpn,blk,pool" 1 0`
