# multilabel-stance-detection-public
Public version of the code for the EMNLP 2019 paper: "Incorporating Label Dependencies in Multilabel Stance Detection" by William Ferreira and Andreas Vlachos.

## Data Preparation
The paper explores methods of performing multilabel stance detection with reference to three multilabel datasets. The datasets cannot be provided with this code due to licensing conditions, and should be requested from the relevant authors:

1. The Brexit Blog Corpus (BBC): Vasiliki  Simaki,  Carita  Paradis,  and  Andreas  Kerren.2017.   Stance classification in texts from blogs onthe 2016 british referendum. In SPECOM.

2. US Election Twitter Corpus (ETC): Parinaz  Sobhani,  Diana  Inkpen,  and  Xiaodan  Zhu.2019.   Exploring  deep  neural  networks  for  multi-target stance detection.Computational Intelligence,35(1):82–97.

3. The Moral Foundation Twitter Corpus (MFTC): Morteza Dehghani, Joseph Hoover, Gwenyth Portillo-Wightman, Leigh Yeh, Shreya Havaldar, Ying Lin,Aida  M  Davani,  Brendan  Kennedy,   MohammadAtari,  Zahra Kamel,  and et al. 2019.   Moral foundations twitter corpus.

The code in this repo is written in Python 3.7.x. To run the code, we suggest you create a Python virtual environment using the Anaconda Python platform (https://www.anaconda.com/), for example:

   <code>conda create --name emnlp_2019_multilabel anaconda</code>
   
In addition to the packages that come with anaconda, you will need:

   <code>conda install -c conda-forge tensorflow-hub</code>
   <code>conda install -c conda-forge keras</code>
   

For each of the datasets, there is an associated script that pre-processes the original dataset to be used in the code:

1. The BBC - run the script <code>prepare_bbc_dataset.py</code> which looks for a file called <code>brexit_blog_corpus.xlsx</code> in the same directory as the script. The script does some data cleaning and pre-processing (tokenizing, generating ELMO embeddings) and saves the output to the same directory. The output consists of the following files:

   a. <code>bbc_dataset.csv</code> - a comma-separated file consisting of an utterance ID, a tokenized utterance string, and binary-valued columns for each of the ten stances, with the obvious interpretation, and a final column indicating whether the utterance is in the 80\% training set, or the 20\% held-out test set.
   
   b. <code>bbc_dataset_folds.csv</code> - a comma-separated file consisting of an utterance ID, and for each cv fold, a column indicating if the utterance is an item in the train or test set for that fold, for all the data in the training set (see a.).
   
   c. <code>bbc_elmo_train_embeddings.csv</code> - a comma-separated file consisting of an utterance ID and the vector representation of the ELMO embedding for the tokenized utterance, for all the utterances in the training set (see a.).
   
   d. <code>bbc_elmo_test_embeddings.csv</code> - a comma-separated file consisting of an utterance ID and the vector representation of the ELMO embedding for the tokenized utterance, for all the utterances in the test set (see a.).
   
2. The ETC - run the script <code>prepare_tweet_dataset.py</code> which looks for a file called <code>all_data_tweet_text.csv</code> in the same directory as the script. The script splits the data in the file into the three target pairs: Donald Trump - Hilary Clinton (DT_HC), Donald Trump - Ted Cruz (DT_TC), and Hilary Clinton - Bernie Sanders (HC_BS). For each target pair the script:

    a. combines the Train and Dev sets to produce a single train set, and keeps the existing test set as a hold-out test set,
    
    b. splits the new training set into five train/test folds,
    
    c. generates a new comma-separated file for each target pair called tweet-x.csv, where x in \{DT_HC, DT_TC, HC_BS\}, the columns are: ID, Tweet, Target 1, Target 2, Test/Train/Dev, set, fold_1, fold_2, fold_3, fold_4, fold_5, and:
    
       i. Tweet is the tokenized tweet,
       ii. Target 1 is the first target stance (e.g. FOR)
       iii. Target 2 is the second target stance (e.g. AGAINST)
       iv. Test/Train/Dev is the original set designator
       v. set is the new set (i.e. train or test) designator
       vi. fold_i indicates whether the instance is in the train or test set for cv fold i in (1..5)
        
    d. generates a comma-separated file consisting of an ID and the vector representation of the ELMO embedding for the tokenized tweet
    
3. The MFTC - run the script <code>prepare_mftc_dataset.py</code> which looks for a file called <code>MFTC_V3_Text.json</code> in the same directory as the script. The script requires an argument <code>--corpus \<corpus name\></code> where <code>corpus name</code> is one of <code>ALM, Baltimore, BLM, Davidson, Election, MeToo</code> or <code>Sandy</code>. For example:
   
      <code>python prepare_mftc_dataset.py --corpus ALM</code>
   
   prepares the data for the ALM dataset of the corpus, and generates two files: <code>moral-dataset-ALM.csv</code> and <code>moral-          dataset-ALM_elmo_embeddings.csv</code>. The first file is comma-separated with columns: ID, Tweet, set, fold_1, fold_2, fold_3, fold_4,    fold_5, where Tweet is the tokenized tweet text, set indicates whether the tweet is in the train or hold-out test set, and fold_i          indicates whether the tweet is in the train or test set for fold i. The second file contains the ELMO embeddings for the tweets,
   keyed by ID. Repeat running the script for the remaining datasets: Baltimore, BLM, ..., Sandy.

## Running the models
The script <code>run_cv.py</code> is used to run the various models against the different datasets. There are various combinations of learning algorithm: FastText (FT), Multi-task Learning (MTL); data encoding methods: Binary-relevance (BR), Label power-set (LP); and loss functions: Binary Cross-entropy, Cross-label dependency (XLD). There are also three datasets, where in addition, the ETC and MFTC datasets are split into sub-datasets. 

The main script parameters are <code>--model-name</code> and <code>--dataset-name</code>, where the model-name and dataset-name can take the following values:

#### --model-name
    mlp-base                     - MTL model with binary cross-entropy loss
    mlp-powerset                 - MTL model with label power-set encoding and categorical cross-entropy loss
    mlp-cross-label-dependency   - MTL model with cross-label dependency loss

    fasttext-binary-relevance    - Fasttext model with binary relevance encoding
    fasttext-powerset            - Fasttext model with label power-set encoding

#### --dataset-name
    bbc              - BBC dataset
    tweets-X         - ETC dataset, X in {DT_HC, DT_TC, HC_BS}
    moral-dataset-X  - MFTC dataset, X in {ALM, BLM, Baltimore, Davidson, Election, MeToo, Sandy}
    
To run a specific model-name/dataset-name combination, for example, mlp-base and bbc run the following command:

    <code>python run_cv.py --model-name mlp-base --dataset-name bbc
    

