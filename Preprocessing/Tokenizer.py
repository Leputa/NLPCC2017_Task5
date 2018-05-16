import jieba

class Tokenizer():

    def replace_line(self,line):
        # 替换为汉语标点
        return line.replace('，', ',').replace('。', '.')\
            .replace('！', '!').replace('？', '?')\
            .replace('“', '"').replace('”', '"')

    def parser(self,line):
        # 汉语分词
        line = self.replace_line(line)
        seg_list = jieba.lcut(line)
        return [word for word in seg_list if word!=" "]
