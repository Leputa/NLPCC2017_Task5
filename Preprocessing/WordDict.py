import pickle
import os
import string
import sys
sys.path.append('../')

from Config import config

class WordDict():
    def __init__(self):
        self.Word2IndexDic = {}
        self.unknown_word = ['<UNKNOWN>', '<DIGITS>', '<ALPHA>', '<ALPHA_NUM>']
        for i, word in enumerate(self.unknown_word):
            self.Word2IndexDic[word] = i

    def check_contain_chinese(self, check_str):
        # 判断字符串中是否有中文字符
        return any(map(lambda ch: u'\u4e00' <= ch <= u'\u9fa5', check_str))


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
            self.Word2IndexDic[word] = self.special_words(word)
        else:
            if self.Word2IndexDic.get(word) == None:
                self.Word2IndexDic[word] = len(self.Word2IndexDic)

    def get_index(self,word):
        return self.Word2IndexDic.get(word)

    def get_size(self):
        return len(self.Word2IndexDic)

    def saveWord2IndexDic(self):
        path = config.cache_prefix_path + "Word2IndexDic.pkl"
        with open(path, 'wb') as pkl:
            pickle.dump(self.Word2IndexDic, pkl)

    def loadWord2IndexDic(self):
        self.Word2IndexDic.clear()
        path = config.cache_prefix_path + 'Word2IndexDic.pkl'
        with open(path, 'rb') as pkl:
            self.Word2IndexDic = pickle.load(pkl)
            return self.Word2IndexDic





