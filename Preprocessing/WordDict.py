import pickle
import os
import string
import sys
sys.path.append('../')

from Config import config


class Dict():
    def __init__(self):
        self.Dic = {}
        self.unknown_word = ['<UNKNOWN>', '<DIGITS>', '<ALPHA>', '<ALPHA_NUM>']
        for i, word in enumerate(self.unknown_word):
            self.Dic[i] = word

    def check_contain_chinese(check_str):
        # 判断字符串中是否有中文字符
        punc = string.punctuation
        return any(map(lambda ch: u'\u4e00' <= ch <= u'\u9fa5' or ch in punc, check_str))


    def special_words(self, word):
        # 处理非中文特殊字符
        if word.isdigit():
            return 1  # '<DIGITS>'
        elif sum([(0 if c in 'qwertyuiopasdfghjklzxcvbnm' else 1) for c in word.lower()])==0:
            return 2  # '<ALPHA>'
        elif sum([(0 if c in 'qwertyuiopasdfghjklzxcvbnm01234567890' else 1) for c in word.lower()])==0:
            return 3  # '<ALPHA_NUM>'
        else:
            return 0  # '<UNKNOWN>'

    def add_word(self, word):
        if self.check_contain_chinese(word) == False:
            self.dic[word] = self.special_words(word)
        else:
            if self.dic.get(word) == None:
                self.dic[word] = len(self.dic)

    def get_index(self,word):
        return self.dic.get(word)

    def get_size(self):
        return len(self.dic)

    def save(self):
        path = config.cache_prefix_path + 'word_dict.pkl'
        with open(path, 'wb') as pkl:
            pickle.dump(self.dic, pkl)

    def load(self):
        self.dic.clear()
        path = config.cache_prefix_path + 'word_dict.pkl'
        with open(path, 'rb') as pkl:
            pickle.load(path, pkl)


