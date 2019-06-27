# ohca_regconize
### requirements of python packages
https://github.com/kunxianhuang/ohca_regconize/blob/master/requirements.txt
* stop words
file "chinese" should put in nltk stopwords
### Training
* command as:

 ``python3 text_ohcarecog_train.py model train ``
* training data is at data/ohca_scripts.txt
### Evaluation 

* command as:

``python3 text_ohcarecog_eval.py --test_file=[TEST_FILE] --outfile=[OUTFILE]`` 

[TEST_FILE] default is data/ohca_test1.txt , you should use only "single" call

[OUTFILE]   defualt is data/ohca_testout.txt
