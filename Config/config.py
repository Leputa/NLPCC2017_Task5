import os

data_prefix_path = '../data/'

TRAIN_FILE = data_prefix_path + 'nlpcc-iccpol-2016.dbqa.training-data'
TEST_FILE = data_prefix_path + 'nlpcc-iccpol-2016.dbqa.testing-data'
WIKI_EMBEDDING_MATRIX = data_prefix_path + 'zhwiki_2017_03.sg_50d.word2vec'

prefix_path = os.getcwd()
prefix_path = prefix_path.replace(prefix_path.split("\\")[-1],"")
cache_prefix_path = prefix_path + 'Cache/'
token_train_pkl = "token_train.pkl"
token_test_pkl = "token_test.pkl"


TOKEN_TRAIN_PKL = cache_prefix_path + token_train_pkl
TOKEN_TEST_PKL = cache_prefix_path + token_test_pkl
IMAGE_PATH = prefix_path + "Image/"