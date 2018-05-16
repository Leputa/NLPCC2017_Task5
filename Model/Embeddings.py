import pickle
import os
import numpy as np
import sys
sys.path.append('../')

from Config import config
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec


class Embeddings():
    def __init__(self):
        self.scale = 0.1
        self.vec_dim = 50

    def get_word_base(self):
        print("合并句子")

        path = config.cache_prefix_path + 'conbine_sentence.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        with open(config.cache_prefix_path + 'token_train.pkl', 'rb') as pkl:
            train_questions, train_answers, train_labels = pickle.load(pkl)
        with open(config.cache_prefix_path + 'train_answer_group.pkl', 'rb') as pkl:
            train_group = pickle.load(pkl)

        train_questions_distinct = []
        i = 0
        for num in train_group:
            train_questions_distinct.append(train_questions[i])
            i += num

        with open(config.cache_prefix_path + 'token_test.pkl', 'rb') as pkl:
            test_questions, test_answers, test_labels = pickle.load(pkl)
        with open(config.cache_prefix_path + 'test_answer_group.pkl', 'rb') as pkl:
            test_group = pickle.load(pkl)

        test_questions_distinct = []
        i = 0
        for num in test_group:
            test_questions_distinct.append(test_questions[i])
            i += num

        combine_sentence = []
        for sentence in [train_questions_distinct, train_answers, test_questions_distinct, test_answers]:
            combine_sentence.extend(sentence)

        with open(path, 'wb') as pkl:
            pickle.dump(combine_sentence, pkl)
        return combine_sentence


    def train_word2vec(self):
        print("训练词向量")

        path = config.cache_prefix_path + 'word2vec_model'
        if os.path.exists(path):
            return Word2Vec.load(path)

        sentences = self.get_word_base()
        model = Word2Vec(sentences, size=self.vec_dim, workers=6, min_count=0)

        model.save(path)
        return model




    def word2vec(self, word_emb, word, scale, vec_dim):
        unknown_word = np.random.uniform(-scale,scale,vec_dim)
        if word in word_emb:
            res = word_emb[word]
            flag = 0
        else:
            res = unknown_word
            flag = 1
        return res,flag

    def wiki_embedding(self):
        print("wiki Embedding!")

        path = config.cache_prefix_path + "wike_index2vec.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word_emb = KeyedVectors.load_word2vec_format(config.WIKI_EMBEDDING_MATRIX)
        with open(config.cache_prefix_path+'Word2IndexDic.pkl','rb') as pkl:
            word2index = pickle.load(pkl)

        vocal_size = len(word2index)
        index2vec = np.zeros((vocal_size + 1, self.vec_dim), dtype="float32")
        index2vec[0] = np.zeros(self.vec_dim)
        unk_count = 0

        for word in word2index:
            index = word2index[word]
            vec, flag = self.word2vec(word_emb, word, self.scale, self.vec_dim)
            index2vec[index] = vec
            unk_count += flag

        print("emb vocab size: ", len(word_emb.vocab))
        print("unknown words count: ", unk_count)
        print("index2vec size: ", len(index2vec))

        with open(path, 'wb') as pkl:
            pickle.dump(index2vec, pkl)
        return index2vec


if __name__ == '__main__':
    embedding = Embeddings()
    model = embedding.train_word2vec()
    print(model)
    embedding.wiki_embedding()
