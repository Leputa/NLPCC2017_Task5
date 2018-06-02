import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.append("../")

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Model.Embeddings import Embeddings


class Rank_CNN():
    def __init__(self):
        self.preprocessor = Preprocess.Preprocessor()
        self.embedding = Embeddings()
        self.lr = 5e-4
        self.batch_size = 128
        self.n_epoch = 10

        self.sentence_length = self.preprocessor.sentence_length
        self.vec_dim = self.embedding.vec_dim
        self.filter_sizes = [2, 3]
        self.num_filters = 64
        self.num_hidden = 100
        self.l2_reg = 0.0004
        self.num_classes = 2

        self.vocab_size = 212237

    def define_model(self):
        self.question = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question')
        self.answer = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')

        with tf.name_scope('embedding'):
            # embedding_matrix = self.embedding.get_wiki_embedding_matrix()
            # embedding_matrix = tf.Variable(embedding_matrix, trainable=True, name='embedding')
            embedding_matrix = tf.get_variable('embedding', [self.vocab_size, self.vec_dim], dtype=tf.float32)
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            question_inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.question),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.question))
            answer_inputs = tf.cond(self.trainable,
                                        lambda: tf.nn.embedding_lookup(embedding_matrix, self.answer),
                                        lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.answer))

            # [batch_size, sentence_length, vec_dim, 1]
            question_expanded = tf.expand_dims(question_inputs, -1)
            answer_expanded = tf.expand_dims(answer_inputs, -1)


        pooled_output_questions = []
        pooled_output_answers = []
        # filter_size: 窗口大小
        for i, filter_size in enumerate(self.filter_sizes):
            #question
            question_conv = self.convolution(question_expanded, self.vec_dim, filter_size, self.num_filters, 'conv-question-')
            question_pooled = self.max_pool(question_conv, filter_size, 'max-pooling-quesiont-')
            pooled_output_questions.append(question_pooled)

            #answer
            answer_conv = self.convolution(answer_expanded, self.vec_dim, filter_size, self.num_filters, 'conv-answer-')
            answer_pooled = self.max_pool(answer_conv, filter_size, 'max-pooling-answer-')
            pooled_output_answers.append(answer_pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        # [batch_size, 1, 1, num_filters] => [batch_size, 1, 1, num_filters_total] => [batch_size, num_filters_total]
        pooling_question = tf.reshape(tf.concat(pooled_output_questions, 3), [-1, num_filters_total], name='pooling_question')
        pooling_answer = tf.reshape(tf.concat(pooled_output_answers, 3), [-1, num_filters_total], name='pooling_answer')
        # [batch_size, 1]
        sims = self.get_sim(pooling_question, pooling_question, num_filters_total)

        self.new_input = tf.concat([pooling_question, sims, pooling_answer], 1, name='new_input')

        with tf.name_scope('hidden'):
            hidden_output = tf.layers.dense(
                inputs = self.new_input,
                units = self.num_hidden,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                name = 'hidden'
                )
            h_drop = tf.nn.dropout(hidden_output, self.dropout_keep_prob, name='hidden_output_drop')

        with tf.name_scope('output'):
            self.scores = tf.layers.dense(
                inputs = h_drop,
                units = self.num_classes,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                bias_initializer = tf.constant_initializer(0.1),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                name = 'scores'
            )

        self.prediction = tf.contrib.layers.softmax(self.scores)[:, 1]

        self.cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        )

        with tf.name_scope('acc'):
            correct = tf.nn.in_top_k(self.scores, self.label, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def get_sim(self, x1, x2, d):
        with tf.name_scope("similarity"):
            M = tf.get_variable(
                name = 'M',
                shape = [d, d],
                initializer = tf.contrib.layers.xavier_initializer()
            )

            x1_trans = tf.matmul(x1, M)
            sims = tf.reduce_sum(tf.multiply(x1_trans, x2), axis=1, keepdims=True)
            return sims



    def max_pool(self, x, filter_size, name_scope):
        with tf.name_scope(name_scope + str(filter_size)):
            pool_width = self.sentence_length - filter_size + 1

            pooled = tf.layers.max_pooling2d(
                inputs= x,
                pool_size= (pool_width, 1),
                strides = 1,
                padding = 'VALID',
            )
            # [batch_size, 1, 1, num_filters]
            return pooled

    def convolution(self, x, d, filter_size, num_filters, name_scope):
        with tf.name_scope(name_scope + str(filter_size)):
            conv = tf.contrib.layers.conv2d(
                inputs = x,
                num_outputs = num_filters,
                kernel_size = (filter_size, d),
                stride = 1,
                padding = 'VALID',
                activation_fn = tf.nn.relu,
                weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                biases_initializer=tf.constant_initializer(0.1),
                trainable = True
            )
            # [batch_size, |s| - filter_size + 1, 1, num_filters]
            return conv

    def train(self):
        save_path = config.save_prefix_path + 'RankCNN/'

        self.define_model()

        (train_questions, train_answers, train_labels) = self.preprocessor.padding_data('train')
        length = len(train_questions)

        (test_questions, test_answers, test_labels) = self.preprocessor.padding_data('test')
        test_questions = np.array(test_questions)
        test_answers = np.array(test_answers)
        test_labels = np.array(test_labels)

        global_steps = tf.Variable(0, name='global_step', trainable=False)

        self.train_op = tf.train.AdamOptimizer(self.lr, name='optimizer').minimize(self.cost, global_step=global_steps)


        # 为了提前停止训练
        # best_loss_test = np.infty
        # checks_since_last_progress = 0
        # max_checks_without_progress = 40
        # best_model_params = None

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            # debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            if os.path.exists(save_path):
                try:
                    ckpt = tf.train.get_checkpoint_state(save_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                except:
                    ckpt = None
            else:
                os.makedirs(save_path)

            for epoch in range(self.n_epoch):
                for iteration in range(length//self.batch_size):
                    train_feed_dict = self.gen_train_dict(iteration, train_questions, train_answers, train_labels, True)
                    train_loss, train_acc, current_step, _ = sess.run([self.cost, self.accuracy, global_steps, self.train_op], feed_dict = train_feed_dict)
                    if current_step % 128 == 0:
                        test_feed_dict = self.gen_test_dict(test_questions, test_answers, test_labels)
                        test_loss = self.cost.eval(feed_dict = test_feed_dict)
                        test_acc = self.accuracy.eval(feed_dict = test_feed_dict)
                        print("**********************************************************************************************************")
                        print("Epoch {}, Iteration {}, train loss: {:.4f}, train accuracy: {:.4f}%.".format(epoch,
                                                                                                            current_step,
                                                                                                            train_loss,
                                                                                                            train_acc * 100))
                        print("Epoch {}, Iteration {}, test loss: {:.4f}, test accuracy: {:.4f}%.".format(epoch,
                                                                                                          current_step,
                                                                                                          test_loss,
                                                                                                          test_acc * 100))
                        print("**********************************************************************************************************")
                        checkpoint_path = os.path.join(save_path, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step = current_step)

            #             if test_loss < best_loss_test:
            #                 best_loss_test = test_loss
            #                 checks_since_last_progress = 0
            #                 best_model_params = tool.get_model_params()
            #             else:
            #                 checks_since_last_progress += 1
            #
            #             if checks_since_last_progress>max_checks_without_progress:
            #                 print("Early Stopping")
            #                 break
            #     if checks_since_last_progress > max_checks_without_progress:
            #         break
            #
            # if best_model_params:
            #     tool.restore_model_params(best_model_params)
            # saver.save(sess, os.path.join(save_path, 'best_model.ckpt'))

    def test(self):
        save_path = config.save_prefix_path + 'RankCNN/'
        assert os.path.isdir(save_path)

        test_questions, test_answers, test_labels = self.preprocessor.padding_data('test')

        tf.reset_default_graph()

        self.define_model()
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            test_results = []
            init.run()
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, os.path.join(save_path, 'best_model.ckpt'))

            for step in tqdm(range(len(test_questions)//self.batch_size + 1)):
                test_feed_dict = self.gen_train_dict(step, test_questions, test_answers, test_labels, False, 1.0)
                pred = sess.run(self.prediction, feed_dict = test_feed_dict)
                test_results.extend(pred.tolist())

        test_group =self.preprocessor.group('test')
        start = 0
        for group_num in test_group:
            tmp_results = test_results[start: start+group_num]
            index = start + tmp_results.index(max(tmp_results))
            test_results[index] = 1
            start += group_num

        with open(config.eval_prefix_path + 'RankCNN-' + 'testing.score.txt', 'w') as fr:
            for result in test_results:
                fr.write(str(result) + '\n')

    def gen_test_dict(self, test_questions, test_answers, test_labels):
        test_index = np.random.randint(0, len(test_questions), [1024])
        test_question_batch = test_questions[test_index]
        test_answer_batch = test_answers[test_index]
        test_label_batch = test_labels[test_index]

        test_feed_dict = {
            self.question: test_question_batch,
            self.answer: test_answer_batch,
            self.label: test_label_batch,
            self.trainable: False,
            self.dropout_keep_prob: 1.0,
        }

        return test_feed_dict

    def gen_train_dict(self, iteration, train_questions, train_answers, train_labels, trainable, keep_prob=0.5):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]
        label_batch = train_labels[start:end]

        feed_dict = {
            self.question: question_batch,
            self.answer: answer_batch,
            self.label: label_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: keep_prob,
        }

        return feed_dict

if __name__ == '__main__':
    rank_cnn = Rank_CNN()
    #rank_cnn.train()
    rank_cnn.test()





