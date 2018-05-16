import pickle
import os
import sys
sys.path.append('../')

from Config import config
from WordDict import *
from Tokenizer import *


class Preprecessor():

    def __init__(self):
        self.train_questions, self.test_questions = [],[]
        self.train_answers, self.test_answers = [],[]
        self.train_labels, self.test_labels = [],[]

    def load_train_data(self):
        print("导入训练数据")

        path = config.cache_prefix_path + config.original_train_pkl
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                self.train_questions, self.train_answers, self.train_labels = pickle.load(pkl)
            return self.train_questions, self.train_answers, self.train_labels

        original_path = config.TRAIN_FILE
        fr = open(original_path,encoding='utf-8')
        lines = fr.readlines()
        for line in lines:
            lineList = line.strip().split("\t")
            self.train_questions.append(lineList[0].strip())
            self.train_answers.append(lineList[1].strip())
            self.train_labels.append(lineList[2].strip())

        with open(path, 'wb') as pkl:
            pickle.dump((self.train_questions,self.train_answers,self.train_labels),pkl)

        return self.train_questions,self.train_answers,self.train_labels

    def load_test_data(self):
        print("导入测试数据")

        path = config.cache_prefix_path + config.original_test_pkl
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                self.test_questions, self.test_answers, self.test_labels = pickle.load(pkl)
            return self.test_questions, self.test_answers, self.test_labels

        original_path = config.TEST_FILE
        fr = open(original_path,encoding='utf-8')
        lines = fr.readlines()
        for line in lines:
            lineList = line.strip().split("\t")
            self.test_questions.append(lineList[0].strip())
            self.test_answers.append(lineList[1].strip())
            self.test_labels.append(lineList[2].strip())

        with open(path, 'wb') as pkl:
            pickle.dump((self.test_questions,self.test_answers,self.test_labels),pkl)

        return self.test_questions,self.test_answers,self.test_labels



if __name__ == '__main__':
    propressor = Preprecessor()
    question,answer,labels = propressor.load_test_data()
    print(question[:10])
    print(answer[:10])
    print(labels[:10])



