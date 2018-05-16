import os

data_prefix_path = '../data/'

TRAIN_FILE = data_prefix_path + 'nlpcc-iccpol-2016.dbqa.training-data'
TEST_FILE = data_prefix_path + 'nlpcc-iccpol-2016.dbqa.testing-data'

prefix_path = os.getcwd()
prefix_path = prefix_path.replace(prefix_path.split("\\")[-1],"")
cache_prefix_path = prefix_path + 'Cache/'
original_train_pkl = "original_train.pkl"
original_test_pkl = "original_test.pkl"


ORIGINAL_TRAIN_PKL = cache_prefix_path + original_train_pkl
ORIGINAL_TEST_PKL = cache_prefix_path + original_test_pkl
IMAGE_PATH = prefix_path + "Image/"