import tensorflow as tf
import numpy as np
import time, sys, os, random
from sklearn import metrics
from sklearn.utils import shuffle
from data_utils import DataUtils
from config import Config
from tsr_models import *


class TrafficSignRecognition:

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.add_model()
        self.add_loss_and_train_op()
        self.add_predict_op()
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
    def load_data(self):
        self.du = DataUtils(self.config.training_file, self.config.testing_file, 
                            self.config.batch_size)
        self.X_train = self.du.train_images  
        self.y_train = self.du.train_labels
        self.X_val = self.du.val_images
        self.y_val = self.du.val_labels
        self.X_test = self.du.test_images
        self.y_test = self.du.test_labels
    
    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.int32)
        self.dropout = tf.placeholder(tf.float32)
    
    def create_feed_dict(self, images, labels=None, dropout=1.):
        feed = {self.inputs: images,
                self.dropout: dropout
               }
        if labels is not None:
            feed[self.labels] = labels
        return feed
                
    def add_model(self):
        self.logits = tsr_model_01(self.inputs, self.dropout)

    def add_loss_and_train_op(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits, self.labels)
        tvars = tf.trainable_variables()
        l2_penalty = tf.add_n([tf.nn.l2_loss(var) for var in tvars])
        self.loss = tf.reduce_mean(loss) + self.config.wd * l2_penalty
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        tf.scalar_summary('loss', self.loss)
        self.summary_op = tf.merge_all_summaries()

    def add_predict_op(self):
        self.prediction = tf.argmax(self.logits, 1)
        
    def run_epoch(self, session, load=None, save=None):
        if load:
            ckpt = tf.train.get_checkpoint_state(load)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            session.run(self.init_op)
        start_time = time.time()
        best_loss = float('inf')
        if not os.path.exists('./save'):
            os.makedirs('./save')
        writer = tf.train.SummaryWriter(self.config.logs_path, graph=tf.get_default_graph())
        for epoch in range(self.config.epochs):
            total_loss = 0
            X_train, y_train = shuffle(self.X_train, self.y_train)
            for i in range(X_train.shape[0]):
                X, y = shuffle(X_train[i], y_train[i])
                feed = self.create_feed_dict(X, y, self.config.dropout)
                loss, _, summary = session.run([self.loss, self.train_op, self.summary_op], 
                                            feed_dict=feed)
                total_loss += loss
                sys.stdout.write('\r')
                sys.stdout.write('\rEpoch {:>3}, step {:>4}, time {:8.2f}, loss {:.4f}'.format(
                    epoch+1, i, time.time()-start_time, total_loss/(i+1)))
                sys.stdout.flush()
            writer.add_summary(summary, epoch*i+1)
            if (epoch+1) % self.config.interval == 0:
                self.evaluation(session)
                if best_loss > total_loss/(i+1):
                    best_loss = total_loss/(i+1)
                    self.saver.save(session, './save/weight_{}'.format(epoch))
            
    def evaluation(self, session, load=None):
        if load:
            ckpt = tf.train.get_checkpoint_state(load)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        train_preds = []
        for i in range(self.X_train.shape[0]):
            feed = self.create_feed_dict(self.X_train[i])
            pred = session.run(self.prediction, feed_dict=feed)
            train_preds.append(pred)
        train_preds = np.vstack(train_preds).flatten()
        train_labels = self.y_train.flatten()
        train_acc = metrics.accuracy_score(train_labels, train_preds)
         
        val_preds = []
        for i in range(self.X_val.shape[0]):
            feed = self.create_feed_dict(self.X_val[i])
            pred = session.run(self.prediction, feed_dict=feed)
            val_preds.append(pred)
        val_preds = np.vstack(val_preds).flatten()
        val_labels = self.y_val.flatten()
        val_acc = metrics.accuracy_score(val_labels, val_preds)
        
        test_preds = []
        for i in range(self.X_test.shape[0]):
            feed = self.create_feed_dict(self.X_test[i])
            pred = session.run(self.prediction, feed_dict=feed)
            test_preds.append(pred)
        test_preds = np.vstack(test_preds).flatten()
        test_labels = self.y_test.flatten()
        test_acc = metrics.accuracy_score(test_labels, test_preds)
        print()
        print('Train {:.4f}, Val {:.4f}, Test {:.4f}'.format(train_acc, val_acc, test_acc))
        
    def predict(self, image, sess, load):
        ckpt = tf.train.get_checkpoint_state(load)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        feed = self.create_feed_dict(image)
        pred = sess.run(self.prediction, feed_dict=feed)
        return pred
                                                
    def get_probabilities(self, image, sess, load):
        ckpt = tf.train.get_checkpoint_state(load)
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        feed = self.create_feed_dict(image)
        probs = sess.run(self.logits, feed_dict=feed)
        return probs


if __name__ == '__main__':
    config = Config()
    tsr = TrafficSignRecognition(config)
    # tf_cfg = tf.ConfigProto()
    # tf_cfg.gpu_options.allow_growth = True
    with tf.Session() as sess:
        tsr.run_epoch(sess)
