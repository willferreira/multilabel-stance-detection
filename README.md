# multilabel-stance-detection-public
Public version of the code for the EMNLP 2019 paper: "Incorporating Label Dependencies in Multilabel Stance Detection" by William Ferreira and Andreas Vlachos.

The paper explores methods of performing multilabel stance detection with reference to three multilabel datasets. The datasets cannot be provided with this code due to licensing conditions, and should be requested from the relevant authors:

1. The Brexit Blog Corpus (BBC): Vasiliki  Simaki,  Carita  Paradis,  and  Andreas  Kerren.2017.   Stance classification in texts from blogs onthe 2016 british referendum. InSPECOM.

2. US Election Twitter Corpus (ETC): Parinaz  Sobhani,  Diana  Inkpen,  and  Xiaodan  Zhu.2019.   Exploring  deep  neural  networks  for  multi-target stance detection.Computational Intelligence,35(1):82–97.

3. The Moral Foundation Twitter Corpus (MFTC): Morteza Dehghani, Joseph Hoover, Gwenyth Portillo-Wightman, Leigh Yeh, Shreya Havaldar, Ying Lin,Aida  M  Davani,  Brendan  Kennedy,   MohammadAtari,  Zahra Kamel,  and et al. 2019.   Moral foun-dations twitter corpus.

For each of the datasets, there is an associated script that pre-processes the original dataset to be used in the code:

1. The BBC - run the script <code>prepare_bbc_dataset.py</code> which looks for a file called <code>brexit_blog_corpus.xlsx</code> in the same directory as the script. The script does some data cleaning and pre-processing (tokenizing, generating ELMO embeddings) and saves the output to the same directory. The output consists of the following files:

   a. <code>bbc_dataset.csv</code> - a comma-separated file consisting of an utterance ID, a tokenized utterance string, and binary-valued columns for each of the ten stances, with the obvious interpretation, and a final column indicating whether the utterance is in the 80\% training set, or the 20\% held-out test set.
   
   b. <code>bbc_dataset_folds.csv</code> - a comma-separated file consisting of an utterance ID, and for each cv fold, a column indicating if the utterance is an item in the train or test set for that fold, for all the data in the training set (see a.).
   
   c. <code>bbc_elmo_train_embeddings.csv</code> - a comma-separated file consisting of an utterance ID and the vector representation of the ELMO embedding for the tokenized utterance, for all the utterances in the training set (see a.).
   
   d. <code>bbc_elmo_test_embeddings.csv</code> - a comma-separated file consisting of an utterance ID and the vector representation of the ELMO embedding for the tokenized utterance, for all the utterances in the test set (see a.).


