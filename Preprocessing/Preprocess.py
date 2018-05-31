import pickle
import os
from tqdm import tqdm
import copy
import gc
import sys
sys.path.append('../')

from Config import config
from Config import langconv
from Config import tool
from Preprocessing.WordDict import *
from Preprocessing.Tokenizer import *


class Preprocessor():

    def __init__(self):
        self.sentence_length = 50

    def load_data_original(self, tag = 'train'):
        print("导入无公共自序列的数据")

        if tag == 'train':
            path = config.cache_prefix_path + 'token_train_original.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'token_test_original.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)


        if tag == 'train':
            original_path = config.TRAIN_FILE
        elif tag == 'test':
            original_path = config.TEST_FILE

        tokenizer = Tokenizer()

        question = []
        answer = []
        label = []


        with open(original_path,encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                lineList = line.strip().split("\t")
                assert len(lineList) == 3
                assert lineList[2] == '0' or lineList[2] == '1'
                # 将问题保存到字典

                # 将繁体字转化为简体
                lineList[0] = langconv.Converter('zh-hans').convert(lineList[0])
                lineList[1] = langconv.Converter('zh-hans').convert(lineList[1])

                # 句子切分
                tmp_quetion = tokenizer.parser(lineList[0].strip())
                tmp_answer = tokenizer.parser(lineList[1].strip())

                question.append(tmp_quetion)
                answer.append(tmp_answer)
                label.append(int(lineList[2]))

        with open(path, 'wb') as pkl:
            pickle.dump((question, answer, label), pkl)

        return (question, answer, label)

    def load_data(self, tag = 'train'):
        print("导入数据")

        if tag == 'train':
            path = config.cache_prefix_path + config.token_train_pkl
        elif tag == 'test':
            path = config.cache_prefix_path + config.token_test_pkl

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)


        if tag == 'train':
            original_path = config.TRAIN_FILE
        elif tag == 'test':
            original_path = config.TEST_FILE

        tokenizer = Tokenizer()

        question = []
        answer = []
        label = []


        with open(original_path,encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                lineList = line.strip().split("\t")
                assert len(lineList) == 3
                assert lineList[2] == '0' or lineList[2] == '1'
                # 将问题保存到字典

                # 将繁体字转化为简体
                lineList[0] = langconv.Converter('zh-hans').convert(lineList[0])
                lineList[1] = langconv.Converter('zh-hans').convert(lineList[1])

                # 句子切分
                tmp_quetion = tokenizer.parser(lineList[0].strip())
                tmp_answer = tokenizer.parser(lineList[1].strip())

                # 添加共同子序列
                common_list = tool.LCS(tmp_quetion, tmp_answer)
                tmp_quetion.extend(common_list)
                tmp_answer.extend(common_list)

                question.append(tmp_quetion)
                answer.append(tmp_answer)
                label.append(int(lineList[2]))

        with open(path, 'wb') as pkl:
            pickle.dump((question, answer, label), pkl)

        return (question, answer, label)


    def word2index(self):
        print("建立词到索引的字典")

        wordDict = WordDict()
        path = config.cache_prefix_path + "Word2IndexDic.pkl"
        if os.path.exists(path):
            return wordDict.loadWord2IndexDic()

        questions, answers, labels  = self.load_data('train')

        for i in range(len(questions)):
            for word in questions[i]:
                wordDict.add_word(word)
            for word in answers[i]:
                wordDict.add_word(word)

        wordDict.saveWord2IndexDic()
        return wordDict.Word2IndexDic

    def get_index_data(self, tag):
        print("将语料转化为索引表示")

        if tag == 'train':
            path = config.cache_prefix_path + 'index_train.pkl'

        elif tag == 'test':
            path = config.cache_prefix_path + 'index_test.pkl'

        if os.path.exists(path) :
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word2IndexDic = self.word2index()

        index_questions = []
        index_answers = []

        (questions, answers, labels) = self.load_data(tag)

        for i in range(len(questions)):
            index_questions.append([word2IndexDic.get(word, 0) for word in questions[i]])
            index_answers.append([word2IndexDic.get(word, 0) for word in answers[i]])

        with open(path, 'wb') as pkl:
            pickle.dump((index_questions, index_answers, labels), pkl)

        return (index_questions, index_answers, labels)

    def padding_data(self, tag):
        print("前向数据padding")

        if tag == 'train':
            path = config.cache_prefix_path + 'padding_train.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'padding_test.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                padding_data = pickle.load(pkl)
            return padding_data

        (index_questions, index_answers, labels) = self.get_index_data(tag)
        padding_questions = copy.deepcopy(index_questions)
        padding_answers = copy.deepcopy(index_answers)
        del index_questions, index_answers
        gc.collect()

        for i in range(len(padding_questions)):
            # questions
            if len(padding_questions[i]) > self.sentence_length:
                padding_questions[i] = padding_questions[i][:self.sentence_length]
            else:
                padding_questions[i] = padding_questions[i] + [0] * (self.sentence_length - len(padding_questions[i]))
            # answers
            if len(padding_answers[i]) > self.sentence_length:
                padding_answers[i] = padding_answers[i][:self.sentence_length]
            else:
                padding_answers[i] = padding_answers[i] + [0] * (self.sentence_length - len(padding_answers[i]))

        with open(path, 'wb') as pkl:
            pickle.dump((padding_questions, padding_answers, labels), pkl)

        return (padding_questions, padding_answers, labels)


    def group(self, tag):
        print("获取每个问题的回答数量")

        if tag == 'train':
            path = config.cache_prefix_path + "train_answer_group.pkl"
        elif tag == 'test':
            path = config.cache_prefix_path + "test_answer_group.pkl"


        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        tmp_question = ""
        group = 0
        groupList = []

        questions, answers, labels = self.load_data_original(tag)

        for i in range(len(questions)):
            question = questions[i]
            print()
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


if __name__ == '__main__':
    preprossor = Preprocessor()
    # preprossor.load_data('train')
    # preprossor.load_data('test')
    # preprossor.word2index()
    # preprossor.get_index_data('train')
    # preprossor.get_index_data('test')
    # preprossor.padding_data('train')
    # preprossor.padding_data('test')
    preprossor.group('train')
    preprossor.group('test')







