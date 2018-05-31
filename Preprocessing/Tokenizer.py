import sys
sys.path.append("../")


import string
from Config import config
import jieba
import jieba.analyse
import jieba.posseg as pseg


class Tokenizer():
    def __init__(self):
        self.stop_words_path = config.data_prefix_path + 'stopwords.txt'
        self.punc = string.punctuation

    def replace_line(self,line):
        # 替换汉语标点为英文标点
        return line.replace('，', ',').replace('。', '.')\
            .replace('！', '!').replace('？', '?')\
            .replace('“', '"').replace('”', '"')

    def parser(self, line):
        # 汉语分词
        line = self.replace_line(line)
        jieba.analyse.set_stop_words(self.stop_words_path)
        words = pseg.cut(line)
        word_list = []

        for w in words:
            word_list.append(w.word)

        return word_list


if __name__ == '__main__':
    tokenizer = Tokenizer()
    wordlist = tokenizer.parser('高富帅是宇宙第一帅')
    print(wordlist)