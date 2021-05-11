import tensorflow as tf
import numpy as np
from collections import namedtuple
from seq2seq_utils import  linear
from tensorflow.python.ops.rnn import dynamic_rnn

HParams=namedtuple("HParams",
                   "nb_users,nb_items,nb_categories,enc_timesteps,topk_categoey,batch_size,hidden_size,min_lr,lr,"
                   "mode,max_grad_norm")
class CategoryModel(object):
    def __init__(self,hps:HParams):
        self._hps=hps
    def run_train_step(self,sess:tf.Session,user_id,_enc_category,real_lenth,target_batch,initial_state):

        to_return = [self._loss, self.topk_cat, self.train_op]
        return sess.run(to_return,
                        feed_dict={
                            self._user_id:user_id,
                            self._enc_category:_enc_category,
                            self._real_lenth: real_lenth,
                            self._target:target_batch,
                            self.initial_state:initial_state,
                        })
    def run_test_step(self,sess:tf.Session,user_id,_enc_category,real_lenth,target_batch,initial_state):
        to_return=[self._loss,self.topk_cat,self._enc_final_output]
        return sess.run(to_return,
                        feed_dict={
                            self._user_id:user_id,
                            self._enc_category:_enc_category,
                            self._real_lenth: real_lenth,
                            self._target:target_batch,
                            self.initial_state:initial_state,
                        })


    def add_placeholders(self):
        hps=self._hps
        self._user_id=tf.placeholder(tf.int32,name="user_id",shape=[hps.batch_size])
        self._enc_category=tf.placeholder(tf.int32,name="_enc_category",shape=[hps.batch_size,hps.enc_timesteps])
        self._real_lenth = tf.placeholder(tf.float32, name="_real_lenth", shape=[hps.batch_size])
        self._target=tf.placeholder(tf.float32,name="_target",shape=[hps.batch_size,hps.nb_categories])

    def create_model(self):
        hps=self._hps
        with tf.variable_scope("CategoryModel"):

            with tf.variable_scope("embedding"):
                user_embedding = tf.get_variable("user_embedding", [hps.nb_users, hps.hidden_size],dtype=tf.float32)
                user_embed=tf.nn.embedding_lookup(user_embedding,self._user_id)
                category_embedding=tf.get_variable("category_embedding",[hps.nb_categories,hps.hidden_size],dtype=tf.float32)
                enc_category_embed=tf.nn.embedding_lookup(category_embedding,self._enc_category)
                enc_category_embed=tf.unstack(enc_category_embed,axis=1)

            #encoder1
            with tf.variable_scope("encoder"):
                encoder_cell=tf.nn.rnn_cell.LSTMCell(hps.hidden_size)
                self.initial_state=encoder_cell.zero_state(batch_size=hps.batch_size,dtype=tf.float32)
                encoder_state=self.initial_state
                encoder_inputs=enc_category_embed
                encoder_outputs=[]

                encoder_inputs_linear=[]
                Matrix=tf.get_variable("cat_user_matrix",[hps.hidden_size*2,hps.hidden_size],initializer=tf.orthogonal_initializer())
                for i, encoder_input in enumerate(encoder_inputs):
                    res= tf.matmul(tf.concat([encoder_input,user_embed],axis=1),Matrix)
                    encoder_inputs_linear.append(res)

                encoder_inputs=tf.expand_dims(tf.concat(encoder_inputs_linear,0),0)

                outputs, states = dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_inputs,
                    sequence_length=self._real_lenth,initial_state=encoder_state)

                self._enc_final_output=states[1]

            with tf.variable_scope("decoder"):

                # Original way.
                # w_1 = tf.get_variable("w_1", initializer=tf.cast(np.zeros([hps.hidden_size, hps.hidden_size]), tf.float32),
                #                       dtype=tf.float32)
                # w_2 = tf.get_variable("w_2", initializer=tf.cast(np.zeros([hps.hidden_size, hps.hidden_size]), tf.float32),
                #                       dtype=tf.float32)
                # w_3= tf.get_variable("w_3", initializer=tf.cast(np.zeros([hps.hidden_size, hps.hidden_size]), tf.float32),
                #                       dtype=tf.float32)
                # category_unstack = tf.unstack(category_embedding, axis=0)
                # x=[]
                # for i, category_input in enumerate(category_unstack):
                #     x.append(tf.matmul(self._enc_final_output,w_1)+tf.matmul([category_input],w_2)+tf.matmul(user_embed,w_3))
                # x=tf.concat(x, axis=0)
                # w=  tf.get_variable("w_softmax", initializer=tf.cast(np.zeros([hps.hidden_size, 2]), tf.float32),
                #                       dtype=tf.float32)
                # b = tf.get_variable("b_category", [hps.nb_categories,2], dtype=tf.float32)
                # y = tf.nn.softmax(tf.matmul(x,w) + b)[:,1]

                # We designed another more efficient way.
                x_1=tf.matmul(user_embed,tf.transpose(category_embedding))
                x_2=tf.matmul(self._enc_final_output,tf.transpose(category_embedding))
                x_3=tf.matmul(tf.nn.embedding_lookup(category_embedding,self._enc_category[:,-1]),tf.transpose(category_embedding))
                x=x_1+x_2+x_3
                b = tf.get_variable("b_category", [hps.nb_categories],dtype=tf.float32)
                y = tf.nn.softmax(x + b,axis=1)

                regularizer_user = tf.contrib.layers.l1_regularizer(0.00001)
                regularizer_cat = tf.contrib.layers.l1_regularizer(0.0001)
                regularizer_matrix = tf.contrib.layers.l1_regularizer(0.001)
                loss_regular_user = tf.contrib.layers.apply_regularization(regularizer_user,
                                                                             weights_list=[user_embedding])
                loss_regular_cat = tf.contrib.layers.apply_regularization(regularizer_cat,
                                                                          weights_list=[category_embedding])
                loss_regular_matrix = tf.contrib.layers.apply_regularization(regularizer_matrix, weights_list=[Matrix,b])

                self._loss = - tf.reduce_sum(tf.log(tf.clip_by_value(y,1e-8,1.0))*self._target)+loss_regular_user+loss_regular_cat+loss_regular_matrix
                _, self.topk_cat = tf.nn.top_k(y, k=hps.topk_categoey)
                # print(self.topk_cat)

    def _add_train_op(self):
        hps = self._hps

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._loss, tvars), hps.max_grad_norm
        )
        # optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=self.global_step)

    def assign_global_step(self, sess: tf.Session, new_value):
        sess.run(tf.assign(self.global_step, new_value))

    def build_graph(self):
        self.add_placeholders()
        self.create_model()
        self.global_step = tf.Variable(0, name="global_step",
                                       trainable=False, dtype=tf.int32)
        if self._hps.mode == "train":
            self._add_train_op()
