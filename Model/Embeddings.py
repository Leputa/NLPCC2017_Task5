import pickle
import os
import numpy as np
import sys
sys.path.append('../')

from Preprocessing import Preprocess

from Config import config
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec


class Embeddings():
    def __init__(self):
        self.scale = 0.1
        self.vec_dim = 50
        self.preprocessor = Preprocess.Preprocessor()

    def get_word_base(self):
        print("合并句子")

        path = config.cache_prefix_path + 'conbine_sentence.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        train_questions, train_answers, train_labels = self.preprocessor.get_train_index_data()
        train_group = self.preprocessor.train_group()

        train_questions_distinct = []
        i = 0
        for num in train_group:
            train_questions_distinct.append(train_questions[i])
            i += num

        test_questions, test_answers, test_labels = self.preprocessor.get_test_index_data()
        test_group = self.preprocessor.test_group()

        test_questions_distinct = []
        i = 0
        for num in test_group:
            test_questions_distinct.append(test_questions[i])
            i += num

        combine_sentence = []
        for sentence in [train_questions_distinct, train_answers, test_questions_distinct, test_answers]:
            combine_sentence.extend(sentence)

        for i in range(len(combine_sentence)):
            combine_sentence[i] = list(map(str, combine_sentence[i]))

        with open(path, 'wb') as pkl:
            pickle.dump(combine_sentence, pkl)
        return combine_sentence


    def train_word2vec(self):

        print("训练词向量")

        path = config.cache_prefix_path + 'word2vec_model'
        if os.path.exists(path):
            return Word2Vec.load(path)

        sentences = self.get_word_base()
        model = Word2Vec(sentences, size=self.vec_dim, window=5, workers=6, sg=1, min_count=1)
        model.save(path)

        return model

    def get_embedding_matrix(self):

        print("embedding")
        path = config.cache_prefix_path + "index2vec.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word_emb = self.train_word2vec()
        word2index = self.preprocessor.word2index()

        vocal_size = len(word2index)
        index2vec = np.zeros((vocal_size + 1, self.vec_dim), dtype="float32")
        index2vec[0] = np.zeros(self.vec_dim)

        for word in word2index:
            index = word2index[word]
            vec, flag = self.word2vec(word_emb, str(index), self.scale, self.vec_dim)
            index2vec[index] = vec

        with open(path, 'wb') as pkl:
            pickle.dump(index2vec, pkl)
        return index2vec

    def word2vec(self, word_emb, word, scale, vec_dim):
        unknown_word = np.random.uniform(-scale,scale,vec_dim)
        if word in word_emb:
            res = word_emb[word]
            flag = 0
        else:
            res = unknown_word
            flag = 1
        return res,flag

    def get_wiki_embedding_matrix(self):
        print("wiki Embedding!")

        path = config.cache_prefix_path + "wike_index2vec.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word_emb = KeyedVectors.load_word2vec_format(config.WIKI_EMBEDDING_MATRIX)
        word2index = self.preprocessor.word2index()

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
    model = embedding.get_embedding_matrix()
    embedding.get_wiki_embedding_matrix()
