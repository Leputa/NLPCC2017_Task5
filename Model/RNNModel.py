import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Embeddings import Embeddings
from layers import MLP, InteractLayer, BatchNormLayer, MLPDropout
import gc
from tqdm import tqdm


class RNNModel():

    def __init__(self):
        self.preprocessor = Preprocess.Preprocessor()
        self.embedding = Embeddings()
        self.sentence_length = self.preprocessor.sentence_length

        self.lr = 0.01
        self.batch_size = 64
        self.n_epoch = 5

        self.n_hidden_rnn = 128
        self.n_out = 2
        self.n_in_mlp = 32
        self.n_hidden_mlp = 128

        self.L1_reg = 0.00
        self.L2_reg = 0.0001

        self.random = False
        self.non_static = False #是否训练embeddings

    def inference(self, embeddings='Leputa', cell='LSTM', tag='forward'):

        with tf.name_scope('placeholder'):
            self.question_inputs = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question_inputs')
            self.answer_inputs = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer_inputs')
            self.label_inputs = tf.placeholder(tf.int32, shape=[None], name='labels_inputs')
            self.keep_prop = tf.placeholder(tf.float32, name='keep_prop')

        question_embed, answer_embed =self.add_embeddings(embeddings)
        question_sentence_vec, answer_sentence_vec = self.add_RNN(question_embed, answer_embed,cell)
        bn_qa_vec = self.add_Interact(question_sentence_vec, answer_sentence_vec)

        pred = self.add_Output(bn_qa_vec)
        accuracy = self.get_eval(pred)

        loss = self.add_loss_op(pred)
        train_op = self.add_train_op(loss)

        return pred, accuracy, loss, train_op

    def train(self, embeddings='Leputa', cell='LSTM', tag='forward'):

        pred, accuracy, loss, train_op = self.inference(embeddings, cell, tag)

        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        if tag == 'forward':
            train_data = self.preprocessor.padding_train_data_forward()
            test_data = self.preprocessor.padding_test_data_forward()
        else:
            train_data = self.preprocessor.padding_train_data_backward()
            test_data = self.preprocessor.padding_test_data_backward()

        length = len(train_data[0])
        test_question = np.array(test_data[0])
        test_answer = np.array(test_data[1])
        test_label = np.array(test_data[2])

        # 为了提前停止训练
        best_loss_test = np.infty
        checks_since_last_progress = 0
        max_checks_without_progress = 20
        best_model_params = None

        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epoch):
                for iteration in range(length//self.batch_size + 1):
                    question_batch, answer_batch, label_batch = self.get_batch(iteration, train_data)
                    sess.run(train_op, feed_dict={self.question_inputs: question_batch,
                                                  self.answer_inputs: answer_batch,
                                                  self.label_inputs: label_batch})
                    if iteration%256 == 0:
                        acc_train = accuracy.eval(feed_dict = {self.question_inputs: question_batch,
                                                               self.answer_inputs: answer_batch,
                                                               self.label_inputs: label_batch})
                        loss_train = loss.eval(feed_dict = {self.question_inputs: question_batch,
                                                               self.answer_inputs: answer_batch,
                                                               self.label_inputs: label_batch})

                        test_question_batch, test_answer_batch, test_label_batch = self.get_test_batch(test_question,test_answer,test_label)
                        acc_test = accuracy.eval(feed_dict={self.question_inputs: test_question_batch,
                                                             self.answer_inputs: test_answer_batch,
                                                             self.label_inputs: test_label_batch})
                        loss_test = loss.eval(feed_dict={self.question_inputs: test_question_batch,
                                                          self.answer_inputs: test_answer_batch,
                                                          self.label_inputs: test_label_batch})


                        print("Epoch {}, Iteration {}, train accuracy: {:.4f}%, test accuracy: {:.4f}%.".format(epoch, iteration, acc_train * 100, acc_test * 100))
                        print("Epoch {}, Iteration {}, train loss: {:.4f}, test loss: {:.4f}.".format(epoch, iteration, loss_train, loss_test))
                        saver.save(sess, config.model_prefix_path + cell + '_model_' + 'tag')

                        if loss_test < best_loss_test:
                            best_loss_test = loss_test
                            checks_since_last_progress = 0
                            best_model_params = tool.get_model_params()
                        else:
                            checks_since_last_progress +=1

                    if checks_since_last_progress > max_checks_without_progress:
                        print("Early Stopping")
                        break
                if checks_since_last_progress > max_checks_without_progress:
                    break

            if best_model_params:
                tool.restore_model_params(best_model_params)

            acc_test = 0
            for step in range(len(test_data[0])//self.batch_size):
                test_question_batch, test_answer_batch, test_label_batch = self.get_batch(step, test_data)
                acc_test += accuracy.eval(feed_dict={self.question_inputs: test_question_batch,
                                                  self.answer_inputs: test_answer_batch,
                                                  self.label_inputs: test_label_batch})
            acc_test = acc_test/(len(test_data[0])//self.batch_size)
            print("best test accuracy: {:.4f}%".format(acc_test * 100))

            save_path = saver.save(sess, config.model_prefix_path + cell + '_model_' + tag + '_best')


    def eval(self, embeddings='Leputa', cell='LSTM', tag='forward'):
        pred, accuracy, loss, train_op = self.inference(embeddings, cell, tag)
        saver = tf.train.Saver()

        if tag == 'forward':
            test_data = self.preprocessor.padding_test_data_forward()
        else:
            test_data = self.preprocessor.padding_test_data_backward()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            test_results = []

            init.run()
            ckpt = tf.train.get_checkpoint_state(config.model_prefix_path + cell + '_model_' + tag + '_best')
            for step in tqdm(range(len(test_data[0])//self.batch_size + 1)):
                test_question_batch, test_answer_batch, test_label_batch = self.get_batch(step, test_data)
                sess.run(pred, feed_dict={self.question_inputs: test_question_batch,
                                          self.answer_inputs: test_answer_batch,
                                          self.label_inputs: test_label_batch})
                batch_pred = pred.eval(feed_dict={self.question_inputs: test_question_batch,
                                                  self.answer_inputs: test_answer_batch,
                                                  self.label_inputs: test_label_batch})
                batch_pred = np.argmax(batch_pred, axis=1)

                test_results.extend(batch_pred.tolist())

        with open(config.eval_prefix_path + 'testing.score.txt','w') as fr:
            for result in test_results:
                fr.write(str(result) + '\n')

    def get_test_batch(self, test_question,test_answer,test_label):
        test_index = np.random.randint(0, len(test_question), [1024])
        return test_question[test_index],test_answer[test_index],test_label[test_index]


    def get_batch(self, iteration, data):
        start = iteration * self.batch_size
        end = min((start+self.batch_size), len(data[0]))
        question_batch = data[0][start:end]
        answer_batch = data[1][start:end]
        label_batch = data[2][start:end]
        return question_batch, answer_batch, label_batch

    def get_eval(self, pred):
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(pred, self.label_inputs, 1)
            accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        return accuracy

    def add_Output(self,bn_qa_vec):
        xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.name_scope("hidden"):
            h = tf.layers.dense(bn_qa_vec, self.n_hidden_mlp, activation=tf.nn.relu, kernel_initializer=xavier_init)
            h_drop = tf.nn.dropout(h, keep_prob=0.5)
        with tf.name_scope("output"):
            pred = tf.layers.dense(h_drop, self.n_out, kernel_initializer=xavier_init)
        return pred

    def add_RNN(self,question_embed,answer_embed,cell):
        with tf.variable_scope("RNN"):
            if cell == 'LSTM':
                rnn_cell = tf.contrib.rnn.LSTMCell(num_units = self.n_hidden_rnn, use_peepholes=True)
            if cell == 'GRU':
                rnn_cell = tf.contrib.rnn.GRUCell(num_units = self.n_hidden_rnn)

            question_outputs, question_states = tf.nn.dynamic_rnn(rnn_cell, inputs=question_embed, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            answer_outputs, answer_states = tf.nn.dynamic_rnn(rnn_cell, inputs=answer_embed, dtype=tf.float32)

            question_sentence_vec = question_outputs[:, -1, :]
            answer_sentence_vec = answer_outputs[:, -1, :]

        return question_sentence_vec, answer_sentence_vec

    def add_Interact(self,question_sentence_vec,answer_sentence_vec):

        with tf.name_scope("InteractLayer"):
            interact_layer = InteractLayer(self.n_hidden_rnn, self.n_hidden_rnn, dim=self.n_in_mlp)
            qa_vec = interact_layer(question_sentence_vec, answer_sentence_vec)
            bn_qa_vec = tf.layers.batch_normalization(qa_vec, momentum=0.9)  #batch_size * n_in_mlp
        return bn_qa_vec

    def add_embeddings(self,embeddings):

        if embeddings == 'Leputa':
            embedding_matrix = self.embedding.get_embedding_matrix()
        elif embeddings == 'wiki':
            embedding_matrix = self.embedding.get_wiki_embedding_matrix()

        with tf.name_scope("embedding"):
            embedding_matrix = tf.Variable(embedding_matrix)
            question_embed = tf.nn.embedding_lookup(embedding_matrix, self.question_inputs)
            answer_embed = tf.nn.embedding_lookup(embedding_matrix, self.answer_inputs)
        return question_embed, answer_embed

    def add_loss_op(self, pred):
        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_inputs, logits=pred)
            loss = tf.reduce_mean(xentropy, name='loss')
        return loss

    def add_train_op(self, loss):
        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op





if __name__ == '__main__':
    model = RNNModel()
    #model.train()
    model.eval()


