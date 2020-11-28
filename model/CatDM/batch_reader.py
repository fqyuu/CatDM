import numpy as np
from collections import defaultdict
from model.CategoryModel import HParams


# HParams = namedtuple("HParams",
#                      "enc_timesteps, dec_timesteps")

class Batcher(object):
    def __init__(self, user_loc_dict, user_loc_dict_label,nb_users, nb_items,nb_times, hps: HParams = None):
        self.user_loc_dict = user_loc_dict
        self.user_loc_dict_label=user_loc_dict_label
        self.nb_users = nb_users
        self.nb_items = nb_items
        self.nb_times = nb_times
        self._hps = hps

    @classmethod
    def from_path(cls, path_train, hps):
        user_loc_dict = defaultdict(list)
        user_loc_dict_label=defaultdict(list)
        all_lenth = np.zeros([hps.nb_users])
        nb_users = 0
        nb_items = 0
        nb_times = 0
        with open(path_train) as f1:
            for line in f1:
                u, i,vc,lat,lon,time,label= line.strip().split(",")
                u, i, vc,label = map(int, [u, i, vc,label])
                # print(u)
                if(label==0):
                    user_loc_dict[u-1].append([i-1,vc-1,1])
                    all_lenth[u - 1] += 1
                else:
                    user_loc_dict_label[u-1].append([i-1, vc-1])
        hps=hps._replace(enc_timesteps=int(max(all_lenth)))
        return cls(user_loc_dict,user_loc_dict_label, nb_users, nb_items, nb_times,hps),int(max(all_lenth))

    def nextBatch_Cat(self, uid):
        hps = self._hps
        check_list_train = np.array(self.user_loc_dict[uid])
        real_lenth=len(self.user_loc_dict[uid])
        if(check_list_train!=[]):
            enc_loc = list(check_list_train[:, 0])
            enc_ve_cat=list(check_list_train[:, 1])
            enc_time= list(check_list_train[:, 2])
            check_list_label = np.array(self.user_loc_dict_label[uid])
            label_pre=list(check_list_label[:, 1])

            if len(enc_ve_cat) < hps.enc_timesteps:
                enc_ve_cat = enc_ve_cat[:]+[0] * (hps.enc_timesteps - len(enc_ve_cat))
                enc_time= enc_time[:]+[0] * (hps.enc_timesteps - len(enc_time))
            else:
                enc_ve_cat=enc_ve_cat[-hps.enc_timesteps:]
                enc_time=enc_time[-hps.enc_timesteps:]

            # label=np.zeros((hps.nb_categories,2),dtype=np.int)
            # for i in range(hps.nb_categories):
            #     if i in label_pre:
            #         label[i][1] = 1
            #     else:
            #         label[i][0]=1

            label=np.zeros((hps.batch_size,hps.nb_categories),dtype=np.int)
            for i in range(hps.nb_categories):
                if i in label_pre:
                    label[0][i]= 1

            return np.array([uid]),\
                    np.array(enc_loc)[np.newaxis,:],\
                    np.array(enc_ve_cat)[np.newaxis,:],\
                   np.array(enc_time), \
                   np.array([real_lenth]), \
                   np.array(label),
        else:
            return None,None,None,None,None,None
