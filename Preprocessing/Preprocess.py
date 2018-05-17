import pickle
import os
import sys
sys.path.append('../')

from Config import config
from Config import langconv
from Preprocessing.WordDict import *
from Preprocessing.Tokenizer import *


class Preprocessor():

    def __init__(self):
        self.sentence_length = 50

        self.train_questions, self.test_questions = [],[]
        self.train_answers, self.test_answers = [],[]
        self.train_labels, self.test_labels = [],[]

        self.train_questions, self.train_answers, self.train_labels = self.load_train_data()
        self.test_questions, self.test_answers, self.test_labels = self.load_test_data()

    def get_train_data(self):
        return self.train_questions, self.train_answers, self.train_labels

    def get_test_data(self):
        return self.test_questions, self.test_answers, self.test_labels

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

    def padding_train_data_forward(self):
        # 填充值在后面
        print("前向训练数据padding")

        path = config.cache_prefix_path + 'train_forward_padding.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        padding_questions, padding_answers, padding_labels = self.get_train_index_data()

        for i in range(len(padding_questions)):
            if len(padding_questions[i]) > self.sentence_length:
                padding_questions[i] = padding_questions[i][:self.sentence_length]
            else:
                pad = [0] * (self.sentence_length - len(padding_questions[i]))
                padding_questions[i] += pad

            if len(padding_answers[i]) > self.sentence_length:
                padding_answers[i] = padding_answers[i][:self.sentence_length]
            else:
                pad = [0] * (self.sentence_length - len(padding_answers[i]))
                padding_answers[i] += pad

        with open(path, 'wb') as pkl:
            pickle.dump((padding_questions, padding_answers, padding_labels),pkl)

        return padding_questions, padding_answers, padding_labels

    def padding_test_data_forward(self):
        # 填充值在后面
        print("前向测试数据padding")

        path = config.cache_prefix_path + 'test_forward_padding.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        padding_questions, padding_answers, padding_labels = self.get_test_index_data()

        for i in range(len(padding_questions)):
            if len(padding_questions[i]) > self.sentence_length:
                padding_questions[i] = padding_questions[i][:self.sentence_length]
            else:
                pad = [0] * (self.sentence_length - len(padding_questions[i]))
                padding_questions[i] += pad

            if len(padding_answers[i]) > self.sentence_length:
                padding_answers[i] = padding_answers[i][:self.sentence_length]
            else:
                pad = [0] * (self.sentence_length - len(padding_answers[i]))
                padding_answers[i] += pad

        with open(path, 'wb') as pkl:
            pickle.dump((padding_questions, padding_answers, padding_labels),pkl)

        return padding_questions, padding_answers, padding_labels

    def padding_train_data_backward(self):
        # 填充值在前面
        print("后向训练数据padding")

        path = config.cache_prefix_path + 'train_backward_padding.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        padding_questions, padding_answers, padding_labels = self.get_train_index_data()

        for i in range(len(padding_questions)):
            if len(padding_questions[i]) > self.sentence_length:
                padding_questions[i] = padding_questions[i][-self.sentence_length:]
            else:
                pad = [0] * (self.sentence_length - len(padding_questions[i]))
                padding_questions[i] = pad + padding_questions[i]

            if len(padding_answers[i]) > self.sentence_length:
                padding_answers[i] = padding_answers[i][-self.sentence_length:]
            else:
                pad = [0] * (self.sentence_length - len(padding_answers[i]))
                padding_answers[i] = pad + padding_answers[i]

        with open(path, 'wb') as pkl:
            pickle.dump((padding_questions, padding_answers, padding_labels),pkl)

        return padding_questions, padding_answers, padding_labels

    def padding_test_data_backward(self):
        # 填充值在前面
        print("后向测试数据padding")

        path = config.cache_prefix_path + 'test_backward_padding.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        padding_questions, padding_answers, padding_labels = self.get_test_index_data()

        for i in range(len(padding_questions)):
            if len(padding_questions[i]) > self.sentence_length:
                padding_questions[i] = padding_questions[i][-self.sentence_length:]
            else:
                pad = [0] * (self.sentence_length - len(padding_questions[i]))
                padding_questions[i] = pad + padding_questions[i]

            if len(padding_answers[i]) > self.sentence_length:
                padding_answers[i] = padding_answers[i][-self.sentence_length:]
            else:
                pad = [0] * (self.sentence_length - len(padding_answers[i]))
                padding_answers[i] = pad + padding_answers[i]

        with open(path, 'wb') as pkl:
            pickle.dump((padding_questions, padding_answers, padding_labels),pkl)

        return padding_questions, padding_answers, padding_labels

if __name__ == '__main__':
    preprossor = Preprocessor()

    preprossor.train_group()
    preprossor.test_group()

    preprossor.padding_train_data_forward()
    preprossor.padding_test_data_forward()

    preprossor.padding_train_data_backward()
    preprossor.padding_test_data_backward()

    word2IndexDic = preprossor.word2index()

    train_index_questions, train_index_answers, train_labels = preprossor.get_train_index_data()
    test_index_questions, test_index_answers, test_labels = preprossor.get_test_index_data()




