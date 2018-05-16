import pickle
import os
import sys
sys.path.append('../')

from Config import config
from Config import langconv
from WordDict import *
from Tokenizer import *


class Preprecessor():

    def __init__(self):
        self.train_questions, self.test_questions = [],[]
        self.train_answers, self.test_answers = [],[]
        self.train_labels, self.test_labels = [],[]

        self.train_questions, self.train_answers, self.train_labels = self.load_train_data()
        self.test_questions, self.test_answers, self.test_labels = self.load_test_data()

    def load_train_data(self):
        print("导入训练数据")

        path = config.cache_prefix_path + config.token_train_pkl
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                self.train_questions, self.train_answers, self.train_labels = pickle.load(pkl)
            return self.train_questions, self.train_answers, self.train_labels

        tokenizer = Tokenizer()
        original_path = config.TRAIN_FILE
        fr = open(original_path,encoding='utf-8')
        lines = fr.readlines()
        for line in lines:
            lineList = line.strip().split("\t")
            # 将繁体字转化为简体
            lineList[0] = langconv.Converter('zh-hans').convert(lineList[0])
            lineList[1] = langconv.Converter('zh-hans').convert(lineList[1])
            self.train_questions.append(tokenizer.parser(lineList[0].strip()))
            self.train_answers.append(tokenizer.parser(lineList[1].strip()))
            self.train_labels.append(int(lineList[2].strip()))

        with open(path, 'wb') as pkl:
            pickle.dump((self.train_questions,self.train_answers,self.train_labels),pkl)

        return self.train_questions,self.train_answers,self.train_labels

    def load_test_data(self):
        print("导入测试数据")

        path = config.cache_prefix_path + config.token_test_pkl
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                self.test_questions, self.test_answers, self.test_labels = pickle.load(pkl)
            return self.test_questions, self.test_answers, self.test_labels

        tokenizer = Tokenizer()
        original_path = config.TEST_FILE
        fr = open(original_path,encoding='utf-8')
        lines = fr.readlines()
        for line in lines:
            lineList = line.strip().split("\t")
            # 将繁体字转化为简体
            lineList[0] = langconv.Converter('zh-hans').convert(lineList[0])
            lineList[1] = langconv.Converter('zh-hans').convert(lineList[1])
            self.test_questions.append(tokenizer.parser(lineList[0].strip()))
            self.test_answers.append(tokenizer.parser(lineList[1].strip()))
            self.test_labels.append(int(lineList[2].strip()))
        fr.close()

        with open(path, 'wb') as pkl:
            pickle.dump((self.test_questions,self.test_answers,self.test_labels),pkl)

        return self.test_questions,self.test_answers,self.test_labels

    def clear_data(self):
        self.train_questions, self.test_questions = [],[]
        self.train_answers, self.test_answers = [],[]
        self.train_labels, self.test_labels = [],[]

    def train_group(self):
        print("获取训练集中每个问题的回答数量")

        path = config.cache_prefix_path + "train_answer_group.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        tmp_question = ""
        group = 0
        groupList = []
        length = len(self.train_questions)

        for i in range(length):
            question = self.train_questions[i]
            if tmp_question == question or tmp_question =="":
                group += 1
                tmp_question = question
            else:
                groupList.append(group)
                group = 1
                tmp_question = question
        groupList.append(group)

        with open(path, 'wb') as pkl:
            pickle.dump(groupList, pkl)

        return groupList


    def test_group(self):
        print("获取测试集中每个问题的回答数量")

        path = config.cache_prefix_path + "test_answer_group.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        tmp_question = ""
        group = 0
        groupList = []
        length = len(self.test_questions)

        for i in range(length):
            question = self.test_questions[i]
            if tmp_question == question or tmp_question =="":
                group += 1
                tmp_question = question
            else:
                groupList.append(group)
                group = 1
                tmp_question = question
        groupList.append(group)

        with open(path, 'wb') as pkl:
            pickle.dump(groupList, pkl)

        return groupList

    def word2index(self):
        print("建立词到索引的字典")
        wordDict = WordDict()
        path = config.cache_prefix_path + "Word2IndexDic.pkl"
        if os.path.exists(path):
            return wordDict.loadWord2IndexDic()

        for i in range(len(self.train_questions)):
            for word in self.train_questions[i]:
                wordDict.add_word(word)
            for word in self.train_answers[i]:
                wordDict.add_word(word)

        for i in range(len(self.test_questions)):
            for word in self.test_questions[i]:
                wordDict.add_word(word)
            for word in self.test_answers[i]:
                wordDict.add_word(word)


        wordDict.saveWord2IndexDic()
        return wordDict.Word2IndexDic

    def get_train_index_data(self):
        print("将训练数据转化为索引表示")

        path = config.cache_prefix_path + "index_train.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word2IndexDic = self.word2index()

        index_questions = []
        index_answers = []

        for i in range(len(self.train_questions)):
            index_questions.append([word2IndexDic[word] for word in self.train_questions[i]])
            index_answers.append([word2IndexDic[word] for word in self.train_answers[i]])

        assert len(index_questions) == len(self.train_questions)
        assert len(index_answers) == len(self.train_answers)
        with open(path, 'wb') as pkl:
            pickle.dump((index_questions,index_answers,self.train_labels),pkl)

        return  index_questions, index_answers, self.train_labels

    def get_test_index_data(self):
        print("将测试数据转化为索引表示")

        path = config.cache_prefix_path + "index_test.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word2IndexDic = self.word2index()

        index_questions = []
        index_answers = []

        for i in range(len(self.test_questions)):
            index_questions.append([word2IndexDic[word] for word in self.test_questions[i]])
            index_answers.append([word2IndexDic[word] for word in self.test_answers[i]])

        assert len(index_questions) == len(self.test_questions)
        assert len(index_answers) == len(self.test_answers)
        with open(path, 'wb') as pkl:
            pickle.dump((index_questions, index_answers, self.test_labels), pkl)

        return index_questions, index_answers, self.test_labels


if __name__ == '__main__':
    propressor = Preprecessor()

    propressor.train_group()
    propressor.test_group()

    word2IndexDic = propressor.word2index()
    train_questions, train_answers, train_labels = propressor.load_train_data()
    train_index_questions, train_index_answers, train_labels = propressor.get_train_index_data()
    test_index_questions, test_index_answers, test_labels = propressor.get_test_index_data()




