import os

data_prefix_path = '../data/'
model_prefix_path = '../Model/model/'
eval_prefix_path = '../evaltool/'

TRAIN_FILE = data_prefix_path + 'nlpcc-iccpol-2016.dbqa.training-data'
TEST_FILE = data_prefix_path + 'nlpcc-iccpol-2016.dbqa.testing-data'
WIKI_EMBEDDING_MATRIX = data_prefix_path + 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'

prefix_path = os.getcwd()
prefix_path = prefix_path.replace(prefix_path.split("\\")[-1],"")
cache_prefix_path = prefix_path + 'Cache/'

token_train_pkl = "token_train.pkl"
token_test_pkl = "token_test.pkl"



IMAGE_PATH = prefix_path + "Image/"