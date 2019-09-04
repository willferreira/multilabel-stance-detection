# multilabel-stance-detection-public
Public version of the code for the EMNLP 2019 paper: "Incorporating Label Dependencies in Multilabel Stance Detection" by William Ferreira and Andreas Vlachos.

The paper explores methods of performing multilabel stance detection wih reference to three multilabel datasets. The datasets are not provided with the code and should be requested from the relevant authors:

1. The Brexit Blog Corpus (BBC): Vasiliki  Simaki,  Carita  Paradis,  and  Andreas  Kerren.2017.   Stance classification in texts from blogs onthe 2016 british referendum. InSPECOM.

2. US Election Twitter Corpus (ETC): Parinaz  Sobhani,  Diana  Inkpen,  and  Xiaodan  Zhu.2019.   Exploring  deep  neural  networks  for  multi-target stance detection.Computational Intelligence,35(1):82–97.

3. The Moral Foundation Twitter Corpus (MFTC): Morteza Dehghani, Joseph Hoover, Gwenyth Portillo-Wightman, Leigh Yeh, Shreya Havaldar, Ying Lin,Aida  M  Davani,  Brendan  Kennedy,   MohammadAtari,  Zahra Kamel,  and et al. 2019.   Moral foun-dations twitter corpus.

For each of the datasets, there is an associated script that pre-processes the original dataset to be used in the code:

1. The BBC - run the script <code>prepare_bbc_dataset.py</code> which looks for a file called <code>brexit_blog_corpus.xlsx</code> in the same directory as the script. The script does some data cleaning and pre-processing (tokenizing, generating ELMO embeddings) and saves the output to the same directory. The output consists of the following files:

   a. <code>bbc_dataset.csv</code> - a comma-separated file consisting of an utterance ID, a tokenized utterance string, and binary-valued columns for each of the ten stances, with the obvious interpretation.
   
   b. <code>


