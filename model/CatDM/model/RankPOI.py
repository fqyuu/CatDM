import tensorflow as tf
import numpy as np
from collections import namedtuple
from seq2seq_utils import  linear
from tensorflow.python.ops.rnn import dynamic_rnn

HParams=namedtuple("HParams",
                   "nb_users,nb_items,nb_times,nb_categories,enc_timesteps,num_windows,batch_size,hidden_size,handset_timesteps,topk_poi,min_lr,lr,activ_prelu,"
                   "mode,max_grad_norm")
class RankPOI(object):
    def __init__(self,hps:HParams):
        self._hps=hps

    def run_train_step(self,sess:tf.Session,user_id,enc_poi,enc_neg_poi,enc_cat,enc_time,enc_windows,real_lenth,poi_category,initial_state):
        to_return = [self._loss,self._poi_rank, self.train_op]
        return sess.run(to_return,
                        feed_dict={
                            self._user_id:user_id,
                            self._enc_poi:enc_poi,
                            self._enc_neg_poi: enc_neg_poi,
                            self._enc_cat: enc_cat,
                            self._enc_time: enc_time,
                            self._enc_windows:enc_windows,
                            self._real_lenth: real_lenth,
                            self._poi_category:poi_category,
                            # self._target:target,
                            self.initial_state:initial_state,
                        })
    def run_test_step(self,sess:tf.Session,user_id,enc_poi,enc_neg_poi,enc_cat,enc_time,enc_windows,real_lenth,poi_category,initial_state):
        to_return = [self._loss, self._poi_rank]
        return sess.run(to_return,
                        feed_dict={
                            self._user_id:user_id,
                            self._enc_poi:enc_poi,
                            self._enc_neg_poi:enc_neg_poi,
                            self._enc_cat: enc_cat,
                            self._enc_time: enc_time,
                            self._enc_windows:enc_windows,
                            self._real_lenth: real_lenth,
                            self._poi_category:poi_category,
                            # self._target:target,
                            self.initial_state:initial_state,
                        })


    # 占位符，数据量少--->batchsize可设为1
    def add_placeholders(self):
        hps=self._hps
        self._user_id=tf.placeholder(tf.int32,name="user_id",shape=[hps.batch_size])
        self._enc_poi=tf.placeholder(tf.int32,name="_enc_poi",shape=[hps.batch_size,hps.enc_timesteps])
        self._enc_neg_poi=tf.placeholder(tf.int32,name="_enc_neg_poi",shape=[hps.batch_size,hps.enc_timesteps])
        self._enc_cat = tf.placeholder(tf.int32, name="_enc_cat", shape=[hps.batch_size, hps.enc_timesteps])
        self._enc_time = tf.placeholder(tf.int32, name="_enc_time", shape=[hps.batch_size, hps.enc_timesteps])
        self._enc_windows= tf.placeholder(tf.float32, name="_enc_windows", shape=[hps.num_windows,hps.enc_timesteps])

        self._poi_category=tf.placeholder(tf.int32,name="_poi_category",shape=[hps.nb_items])
        self._target=tf.placeholder(tf.float32,name="_target",shape=[hps.batch_size,hps.nb_items])
        self._real_lenth = tf.placeholder(tf.int32, name="_real_lenth", shape=[hps.batch_size])

        # self._preference_cat = tf.placeholder(tf.float32,name="_preference_cat",shape=[hps.batch_size,hps.hidden_size])

    # 建模
    def create_model(self):
        hps=self._hps
        with tf.variable_scope("RankPOI"):

            with tf.variable_scope("embedding"):
                user_embedding = tf.get_variable("user_embedding", [hps.nb_users, hps.hidden_size],dtype=tf.float32)
                user_embed=tf.nn.embedding_lookup(user_embedding,self._user_id)

                poi_embedding = tf.get_variable("poi_embedding", [hps.nb_items, hps.hidden_size],dtype=tf.float32)
                enc_poi_embed = tf.nn.embedding_lookup(poi_embedding, self._enc_poi)
                enc_poi_embed=tf.unstack(enc_poi_embed, axis=1)

                category_embedding = tf.get_variable("category_embedding", [hps.nb_categories, hps.hidden_size],dtype=tf.float32, trainable=True)
                enc_cat_embed = tf.nn.embedding_lookup(category_embedding, self._enc_cat)

                time_embedding = tf.get_variable("time_embedding", [hps.nb_times, hps.hidden_size], dtype=tf.float32)
                enc_time_embed = tf.nn.embedding_lookup(time_embedding, self._enc_time)

            #encoder2
            with tf.variable_scope("encoder"):
                encoder_cell=tf.nn.rnn_cell.LSTMCell(hps.hidden_size)
                self.initial_state=encoder_cell.zero_state(batch_size=1,dtype=tf.float32)
                encoder_state=self.initial_state
                encoder_inputs=enc_poi_embed

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

                outputs_temp=outputs[0]
                outputs_trans = tf.transpose(outputs_temp)
                concat_windows_state=[]
                all_w_windows=[]
                for i in range(hps.num_windows):
                    window=tf.expand_dims(self._enc_windows[i],0)
                    result = tf.multiply(outputs_trans, window)
                    result=tf.transpose(result)
                    x=tf.reduce_sum(result,0,keepdims=True)
                    y=tf.reduce_sum(window)
                    y=tf.clip_by_value(y, 1.0, hps.enc_timesteps)
                    window_state=tf.divide(x,y)
                    w_window =linear([tf.nn.leaky_relu(tf.concat([user_embed,tf.multiply(window_state,user_embed),window_state],axis=1), alpha=0.0, name=None)],1,bias=True,scope="w_{}_linear".format(i))
                    all_w_windows.append(w_window)
                    concat_windows_state.append(tf.multiply(window_state,w_window))

                concat_windows_state= tf.reduce_sum(concat_windows_state, 0)
                all_w_windows=tf.reduce_sum(all_w_windows, 0)
                window_state = tf.divide(concat_windows_state, all_w_windows)

                # self.linear_windows = linear([window_state,user_embed], hps.hidden_size, bias=False,scope="window_linear")

                # self.linear_windows=tf.layers.batch_normalization(self.linear_windows, training=True, name="bn_linear_windows")
                # self.concat_windows_state=tf.concat(concat_windows_state,axis=1)

            with tf.variable_scope("decoder"):
                b = tf.get_variable("b_poi", [hps.nb_items],dtype=tf.float32)
                sequence_loss=0
                for i in range(hps.batch_size):
                    loss_list = []
                    for j in range(hps.enc_timesteps-1):
                        pos_p = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_poi[i, j+1]) - tf.nn.embedding_lookup(user_embedding, self._user_id[i])) ** 2
                        pos_t = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_poi[i, j+1])-tf.nn.embedding_lookup(time_embedding, self._enc_time[i, j])) ** 2
                        pos_s = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_poi[i, j+1]) -outputs[i,j] ) ** 2
                        pos_c = tf.norm(tf.nn.embedding_lookup(category_embedding, tf.nn.embedding_lookup(self._poi_category,self._enc_poi[i,j+1]))
                                        - tf.nn.embedding_lookup(user_embedding, self._user_id[i])) ** 2
                        pos_prob = pos_p + pos_s + pos_t+pos_c + b[self._enc_poi[i, j+1]]

                        neg_p = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[i, j+1]) - tf.nn.embedding_lookup(user_embedding, self._user_id[i])) ** 2
                        neg_t = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[i, j+1])-tf.nn.embedding_lookup(time_embedding, self._enc_time[i, j])) ** 2
                        neg_s = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[i, j+1])- outputs[i,j] ) ** 2
                        neg_c = tf.norm(tf.nn.embedding_lookup(category_embedding, tf.nn.embedding_lookup(self._poi_category,self._enc_neg_poi[i,j+1]))
                                        - tf.nn.embedding_lookup(user_embedding, self._user_id[i])) ** 2
                        neg_prob = neg_p + neg_s + neg_t+neg_c + b[self._enc_neg_poi[i, j+1]]
                        p_sub = pos_prob-neg_prob
                        loss_list.append(-tf.log(tf.divide(1, 1 + tf.exp(p_sub))))
                    sequence_loss += tf.reduce_sum(tf.stack(loss_list, axis=0)[0:self._real_lenth[i]])

                all_batch = []
                for i in range(hps.batch_size):
                    pos_p_rank = tf.reduce_sum((poi_embedding - tf.nn.embedding_lookup(user_embedding, self._user_id[i])) ** 2, axis=1)
                    pos_t_rank = tf.reduce_sum((poi_embedding- tf.nn.embedding_lookup(time_embedding, self._enc_time[i, -1])) ** 2,axis=1)
                    pos_s_rank = tf.reduce_sum((poi_embedding - outputs[i,-1]) ** 2, axis=1)
                    pos_win_rank = tf.reduce_sum((poi_embedding - window_state[i]) ** 2, axis=1)
                    pos_c_rank = tf.reduce_sum((tf.nn.embedding_lookup(category_embedding, self._poi_category) - tf.nn.embedding_lookup(user_embedding, self._user_id[i])) ** 2, axis=1)
                    p_next_all_poi = pos_p_rank + pos_t_rank + pos_s_rank +pos_win_rank+ pos_c_rank+b
                    all_batch.append(p_next_all_poi)

                all_batch_stack=tf.stack(all_batch, axis=0)
                _, self._poi_rank = tf.nn.top_k(all_batch_stack*-1,hps.nb_items)
                pos_final_pro=tf.reduce_sum(all_batch_stack*self._target,axis=1)

                neg__final_p = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[0, -1])
                                       - tf.nn.embedding_lookup(user_embedding, self._user_id[0])) ** 2
                neg__final_t = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[0, -1])
                                       - tf.nn.embedding_lookup( time_embedding, self._enc_time[0, -1])) ** 2
                neg__final_s = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[0, -1]) - outputs[0, -1]) ** 2
                neg_win_rank = tf.norm(tf.nn.embedding_lookup(poi_embedding, self._enc_neg_poi[0, -1]) - window_state[0]) ** 2
                neg__final_c = tf.norm(tf.nn.embedding_lookup(category_embedding, tf.nn.embedding_lookup(self._poi_category, self._enc_neg_poi[0, -1]))
                                - tf.nn.embedding_lookup(user_embedding, self._user_id[0])) ** 2
                neg_prob_final = neg__final_p + neg__final_t + neg__final_s+neg_win_rank +neg__final_c + b[self._enc_neg_poi[0, -1]]

                regularizer_user = tf.contrib.layers.l1_regularizer(0.0001)
                regularizer_poi = tf.contrib.layers.l1_regularizer(0.00001)
                regularizer_matrix = tf.contrib.layers.l1_regularizer(0.001)
                loss_regular_user = tf.contrib.layers.apply_regularization(regularizer_user,
                                                                             weights_list=[user_embedding])
                loss_regular_poi = tf.contrib.layers.apply_regularization(regularizer_poi,
                                                                          weights_list=[poi_embedding,category_embedding])
                loss_regular_matrix = tf.contrib.layers.apply_regularization(regularizer_matrix, weights_list=[Matrix,b])
                self._loss = sequence_loss+loss_regular_user+loss_regular_poi+loss_regular_matrix


    def _add_train_op(self):
        self.train_op = tf.train.AdamOptimizer().minimize(self._loss)
    def assign_global_step(self, sess: tf.Session, new_value):
        sess.run(tf.assign(self.global_step, new_value))

    def build_graph(self):
        self.add_placeholders()
        self.create_model()
        self.global_step = tf.Variable(0, name="global_step",
                                       trainable=False, dtype=tf.int32)
        if self._hps.mode == "train":
            self._add_train_op()

