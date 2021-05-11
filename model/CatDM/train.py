import tensorflow as tf
from model.CategoryModel import HParams, CategoryModel
from batch_reader import Batcher
import numpy as np
import os, sys
from FilterPOI import filter_to_generate
import pickle
from collections import defaultdict

tf.flags.DEFINE_string("train_or_test", "train", "Value can be selected by train or test")
tf.flags.DEFINE_string("data_type", "NYC",
                       "the type of dataset, NYC or TKY")
FLAGS = tf.flags.FLAGS

train_data_path="../../data/Foursquare/{}/{}_TRAIN_SPLIT.csv".format(FLAGS.data_type,FLAGS.data_type)
valid_data_path="../../data/Foursquare/{}/{}_VALID_SPLIT.csv".format(FLAGS.data_type,FLAGS.data_type)
test_data_path="../../data/Foursquare/{}/{}_TEST_SPLIT.csv".format(FLAGS.data_type,FLAGS.data_type)

read_POI_path="../../data/Foursquare/{}/{}_VENUE_CAT_LON_LAT.csv".format(FLAGS.data_type,FLAGS.data_type)
write_filterPOI_path="../../data/Foursquare/{}/{}_FILTER_POI.txt".format(FLAGS.data_type,FLAGS.data_type)
prestate_save_path="result/pre_train.pkl"

save_dir = "./save/CategoryModel/{}".format(FLAGS.data_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir_best = os.path.join(save_dir, "best")

# short_state_path = "./short_state.pkl"

# if os.path.exists(short_state_path):
#     print("loading the short_state")
#     with open(short_state_path, "rb") as f:
#         short_state = pickle.load(f)
# else:
#     short_state = defaultdict()


def view_bar(num, total):
    rate = num / total
    rate_num = int(rate * 100)
    r = "\r 当前进度 \t {}%".format(rate_num)
    sys.stdout.write(r)
    sys.stdout.flush()

def _train(hps, batcher):
    start_global_step = 1
    with tf.Graph().as_default(), tf.Session() as sess,tf.device("/cpu:0"):
        with tf.variable_scope("Model"):
            train_model = CategoryModel(hps)
            train_model.build_graph()
            # writer = tf.summary.FileWriter("./graph", sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)

            if os.path.exists(os.path.join(save_dir, "checkpoint")):
                print("continue to train")
                ckpt = tf.train.get_checkpoint_state(save_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                start_global_step = int(ckpt.model_checkpoint_path.split("-")[-1]) + 1

            print("start to train")
            for step in range(start_global_step, 30):
                # train_model.assign_global_step(sess, step)
                losses = 0
                count = 0
                all_rec = float(0)

                for user in np.random.permutation(hps.nb_users):
                    # print("user: ",user)
                    state = sess.run(train_model.initial_state)
                    batch_user,current_poi, enc_ve_cat, enc_time,real_lenth,target_batch= batcher.nextBatch_Cat(user)
                    if(enc_ve_cat is not None):
                        loss,topk_cat ,_= train_model.run_train_step(
                                sess,batch_user,enc_ve_cat,real_lenth,target_batch, state)
                        losses += loss
                        count += 1
                        _,rec=compute_pre_rec_one(topk_cat,target_batch,hps.topk_categoey)
                        all_rec += float(rec)

                print("step:{},train_loss:{:.5f},train_rec:{:.5f}".format(step, losses / count, all_rec / count))

                checkpoint_path = os.path.join(save_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                # if need
                # _valid(hps,step)
                _valid(hps,step)
                hps = hps._replace(mode="train")

def _valid(hps,step):
    # hps = hps._replace(enc_timesteps=40)
    batcher_test,enc_timesteps = Batcher.from_path(valid_data_path,hps)
    hps = hps._replace(mode="train",enc_timesteps=enc_timesteps)
    ckpt = tf.train.get_checkpoint_state(save_dir)

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model"):

            test_model = CategoryModel(hps)
            test_model.build_graph()
            # writer = tf.summary.FileWriter("./graph", sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # choose
            # saver.restore(sess, os.path.join(save_dir, "model.ckpt-18"))

            losses = 0
            count = 0
            all_rec = float(0)
            all_pre_5 = float(0)
            all_rec_5 = float(0)
            all_pre_10 = float(0)
            all_rec_10 = float(0)
            all_pre_15 = float(0)
            all_rec_15 = float(0)
            # print("start to test")
            pre_user_state_dict = defaultdict(list)
            for user in range(hps.nb_users):
                state = sess.run(test_model.initial_state)
                batch_user,current_poi, enc_ve_cat, enc_time,real_lenth ,target_batch = batcher_test.nextBatch_Cat(user)
                if (enc_ve_cat is not None):
                    loss, topk_cat,final_output = test_model.run_train_step(
                        sess, batch_user, enc_ve_cat, real_lenth,target_batch, state)

                    losses += loss
                    count += 1
                    _,rec_z=compute_pre_rec_one(topk_cat,target_batch,hps.topk_categoey)
                    pre_5, rec_5 = compute_pre_rec_one(topk_cat,target_batch,5)
                    pre_10, rec_10 = compute_pre_rec_one(topk_cat,target_batch,10)
                    pre_15, rec_15 = compute_pre_rec_one(topk_cat,target_batch,15)
                    all_pre_5 += float(pre_5)
                    all_rec_5 += float(rec_5)
                    all_pre_10 += float(pre_10)
                    all_rec_10 += float(rec_10)
                    all_pre_15 += float(pre_15)
                    all_rec_15 += float(rec_15)
                    all_rec +=float(rec_z)
                    # pre_user_state_dict[user].append(final_output)

            print("step:{}, valid_loss: {:.5f},valid_pre5:{:.5f}, valid_rec5:{:.5f}, valid_pre10:{:.5f}, valid_rec10:{:.5f}, valid_pre15:{:.5f}, valid_rec15:{:.5f}, all_rec:{:.5f}"
                .format(step, losses / count, all_pre_5 / count, all_rec_5 / count, all_pre_10 / count,
                        all_rec_10 / count, all_pre_15 / count, all_rec_15 / count, all_rec / count))

            # _valid(hps,step)
            _test(hps, step)

def _test(hps,step):
    batcher_test,enc_timesteps= Batcher.from_path(test_data_path,hps)
    hps = hps._replace(mode="test",enc_timesteps=enc_timesteps)

    ckpt = tf.train.get_checkpoint_state(save_dir)

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Model"),tf.device("/cpu:0"):

            test_model = CategoryModel(hps)
            test_model.build_graph()
            # writer = tf.summary.FileWriter("./graph", sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, os.path.join(save_dir, "model.ckpt-25"))

            losses = 0
            count = 0
            all_rec = float(0)
            all_pre_5 = float(0)
            all_rec_5 = float(0)
            all_pre_10 = float(0)
            all_rec_10 = float(0)
            all_pre_15 = float(0)
            all_rec_15 = float(0)
            # print("start to test")
            pre_user_state_dict = defaultdict(list)
            if not os.path.exists(write_filterPOI_path):
                file_t = open(write_filterPOI_path, 'w')
                file_t.close()
            with open(write_filterPOI_path, 'r+') as file:
                file.truncate(0)
            for user in range(hps.nb_users):
                state = sess.run(test_model.initial_state)
                batch_user,current_poi, enc_ve_cat, enc_time,real_lenth ,target_batch = batcher_test.nextBatch_Cat(user)
                if (enc_ve_cat is not None):
                    loss, topk_cat,final_output = test_model.run_test_step(
                        sess, batch_user, enc_ve_cat, real_lenth,target_batch, state)

                    filter_to_generate(user, topk_cat, current_poi[0][-1], read_POI_path, write_filterPOI_path)
                    losses += loss
                    count += 1
                    _,rec_z=compute_pre_rec_one(topk_cat,target_batch,hps.topk_categoey)
                    pre_5, rec_5 = compute_pre_rec_one(topk_cat,target_batch,5)
                    pre_10, rec_10 = compute_pre_rec_one(topk_cat,target_batch,10)
                    pre_15, rec_15 = compute_pre_rec_one(topk_cat,target_batch,15)
                    all_pre_5 += float(pre_5)
                    all_rec_5 += float(rec_5)
                    all_pre_10 += float(pre_10)
                    all_rec_10 += float(rec_10)
                    all_pre_15 += float(pre_15)
                    all_rec_15 += float(rec_15)
                    all_rec +=float(rec_z)
                    # pre_user_state_dict[user].append(final_output)

            print("step:{}, test_loss: {:.5f}, test_pre5:{:.5f}, test_rec5:{:.5f}, test_pre10:{:.5f}, test_rec10:{:.5f}, test_pre15:{:.5f}, test_rec15:{:.5f}, all_rec:{:.5f}"
                .format(step, losses / count, all_pre_5 / count, all_rec_5 / count, all_pre_10 / count,
                        all_rec_10 / count, all_pre_15 / count, all_rec_15 / count, all_rec / count))


def compute_pre_topkcat_one(topk_cat,target_batch):
    all_pre = 0
    count_possitive_real = 0
    count_possitive_predict = 0
    for i, j in enumerate(target_batch):
        if j[1] == 1:
            count_possitive_real += 1
            if i in topk_cat:
                count_possitive_predict += 1

    all_pre += count_possitive_predict / count_possitive_real
    return format(float(all_pre),".5f")

def compute_pre_rec_one(topk_cat,target_batch,top_k):
    topk_cat=topk_cat[0][0:top_k]
    target_batch=target_batch[0]
    count=0
    for i in range (len(topk_cat)):
        if target_batch[topk_cat[i]]==1:
           count += 1
    pre=format(float(count) / float(top_k),".5f")
    rec=format(float(count) / float(np.sum(target_batch)),".5f")
    return pre,rec

def main(_):
    hps = HParams(
        nb_users=1083,
        nb_items=9989,
        nb_categories=233,
        topk_categoey=130,
        batch_size=1,
        min_lr=0.0001,
        lr=0.001,
        hidden_size=64,
        enc_timesteps=0,
        mode="train",
        max_grad_norm=5,
    )
    batcher,enc_timesteps= Batcher.from_path(train_data_path, hps)
    hps=hps._replace(enc_timesteps=enc_timesteps)

    if FLAGS.train_or_test == "train":
        _train(hps, batcher)
    if FLAGS.train_or_test == "test":
        _test(hps,0)

if __name__ == '__main__':
    tf.app.run()
