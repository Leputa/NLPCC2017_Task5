import sys
sys.path.append("../")

import jieba
import string
from Config import config



class Tokenizer():
    def __init__(self):
        self.stopwords = self.get_stop_words()
        self.punc = string.punctuation

    def replace_line(self,line):
        # 替换汉语标点为英文标点
        return line.replace('，', ',').replace('。', '.')\
            .replace('！', '!').replace('？', '?')\
            .replace('“', '"').replace('”', '"')

    def get_stop_words(self):
        stopWords = []
        with open(config.data_prefix_path + 'stopwords.txt','r',encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line.strip() != '':
                    stopWords.append(line.strip())
        return stopWords

    def parser(self,line):
        # 汉语分词
        line = self.replace_line(line)
        seg_list = jieba.lcut(line)

        return [word for word in seg_list if word!=" " and word not in self.punc and word not in self.stopwords]


if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(tokenizer.get_stop_words())