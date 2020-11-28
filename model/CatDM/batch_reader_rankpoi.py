import numpy as np

from collections import defaultdict
from model.RankPOI import HParams
import datetime
import random

# HParams = namedtuple("HParams",
#                      "enc_timesteps, dec_timesteps")

class Batcher(object):
    def __init__(self, user_loc_dict, user_loc_dict_label,user_time_dict,user_candidate_dict,time_windows,user_assaada,test_users, hps: HParams = None):
        self.user_loc_dict = user_loc_dict
        self.user_loc_dict_label=user_loc_dict_label
        self.user_time_dict=user_time_dict
        self.user_candidate_dict=user_candidate_dict
        self.time_windows=time_windows
        self._hps = hps
        self.user_assaada=user_assaada
        self.test_users=test_users

    @classmethod
    def from_path_with_time(cls, path_train,path_test,path_candidate,alldata_path, hps):
        user_loc_dict = defaultdict(list)
        user_time_dict = defaultdict(list)
        user_loc_dict_label=defaultdict(list)
        user_candidate_dict=defaultdict(list)
        user_assaada=defaultdict(list)
        all_lenth = np.zeros([hps.nb_users])
        with open(path_train) as f1:
            for line in f1:
                u, i,vc,lat,lon,time,label= line.strip().split(",")
                u, i, vc,label = map(int, [u, i, vc,label])
                # print(u)
                if(label==0):
                    user_loc_dict[u-1].append([i-1,vc-1])
                    all_lenth[u - 1] += 1
                    user_time_dict[u-1].append(time)
                # else:
                    # user_loc_dict_label[u-1].append([i-1, vc-1])
        if hps.handset_timesteps==0:
            hps = hps._replace(enc_timesteps=int(max(all_lenth)))

        test_users=[]
        with open(path_test) as f2:
            for line in f2:
                u,i,_,_,_,_,label= line.strip().split(",")
                u,i ,label= map(int, [u,i,label])
                if (label==0):
                    if u-1 not in test_users:
                        test_users.append(u-1)
                if (label==1):
                    user_loc_dict_label[u-1].append([i-1])

        with open(path_candidate) as f1:
            for line in f1:
                u, i, vc= line.strip().split(",")
                u, i, vc = map(int, [u, i, vc])
                user_candidate_dict[u].append([i,vc])

        ####################################################################
        minut_count = []
        for i in range(1440):
            minut_count.append(0)
        with open(alldata_path) as f1:
            for line in f1:
                _, _, _, _, _, time = line.strip().split(",")
                minut_count[int(time[-8:-6]) * 60 + int(time[-5:-3])] += 1
        sum = np.sum(minut_count)
        minut_pro = np.divide(minut_count, sum)
        time_windows = []
        probability = float(0)
        time_windows.append(0)
        for i, line in enumerate(minut_pro):
            probability += line
            if (probability >= (1 / 12)):
                probability = float(0)
                time_windows.append(i)
        time_windows.append(1440)
        ####################################################################

        ####################################################################
        # time_windows = []
        # time_windows.append(0)
        # for i in range(12):
        #     time_windows.append(i*120)
        # time_windows.append(1440)
        ####################################################################
        with open("../../data/Foursquare/NYC/NYC_VENUE_CAT_LON_LAT.csv") as f1:
            for line in f1:
                i,c,_,_= line.strip().split(",")
                i,c = map(int, [i, c])
                user_assaada[0].append([i-1,c-1])

        return cls(user_loc_dict, user_loc_dict_label,user_time_dict, user_candidate_dict,time_windows,user_assaada,test_users, hps)

    def nextBatch_POI_Train(self, uid):
        hps = self._hps
        check_list_train = np.array(self.user_loc_dict[uid])
        check_list_candidate = np.array(self.user_candidate_dict[uid])
        user_assaada_li=np.array(self.user_assaada[0])

        real_lenth = len(self.user_loc_dict[uid])
        if real_lenth > hps.enc_timesteps:
            real_lenth = hps.enc_timesteps

        if len(check_list_train) != 0:
            enc_loc = list(check_list_train[:, 0])
            enc_cat=list(check_list_train[:, 1])
            if len(enc_loc) < hps.enc_timesteps:
                enc_loc = enc_loc[:]+[0] * (hps.enc_timesteps - len(enc_loc))
                enc_cat = enc_cat[:]+[0] * (hps.enc_timesteps - len(enc_cat))
            else:
                enc_loc=enc_loc[-hps.enc_timesteps:]
                enc_cat = enc_cat[-hps.enc_timesteps:]
                # enc_time=enc_time[-hps.enc_timesteps:]
            enc_time = np.zeros((hps.enc_timesteps), dtype=np.int32)

            enc_windows=np.zeros(shape=[12,hps.enc_timesteps],dtype=int)
            check_list_time=self.user_time_dict[uid]
            for i in range(12):
                for j in range(real_lenth):
                    time=check_list_time[j]
                    timepoint=int(time[-8:-6])*60+int(time[-5:-3])
                    if timepoint>=self.time_windows[i] and timepoint<self.time_windows[i+1]:
                        enc_windows[i][j]=1
                        enc_time[j]=i

            for i in range(real_lenth):
                cur_date = datetime.datetime.strptime(check_list_time[i], '%Y-%m-%d %H:%M:%S')
                cur_week_num= self.week_number(cur_date.weekday())
                enc_time[i] =enc_time[i] + cur_week_num

            enc_neg_poi=np.zeros((hps.enc_timesteps),dtype=np.int32)
            for i in range(hps.enc_timesteps):
                enc_neg_poi[i]=random.sample(set(range(0,hps.nb_items))^set(check_list_train[:, 0]),1)[0]

            poi_category = list(user_assaada_li[:, 1])

            check_list_label = np.array(self.user_loc_dict_label[uid])
            label = np.zeros((hps.nb_items), dtype=np.int)
            if len(check_list_label)!=0:
                label_pre=list(check_list_label[:, 0])
                for i in range(hps.nb_items):
                    if i in label_pre:
                        label[i] = 1

            poi_candidate = []
            if len(check_list_candidate)!=0:
                poi_candidate = list(check_list_candidate[:, 0])

            is_real=0
            if uid in self.test_users:
                is_real=1
            # if lenth_cadidate!=0:
            #     clf_poi = list(user_assaada_li[:, 0])
            #     clf_category = list(user_assaada_li[:, 1])
            #     label = np.zeros((hps.batch_size,hps.nb_items), dtype=np.int)
            #     for i in range(hps.nb_items):
            #         if i in label_pre:
            #             label[0][i]= 1
            #
            # else:
            #     clf_poi = [0] * (hps.nb_items)
            #     clf_category = [0] * (hps.nb_items )
            #     label = np.zeros((hps.nb_items, 2), dtype=np.int)


            return [uid],\
                    np.array(enc_loc)[np.newaxis,:], \
                   np.array(enc_neg_poi)[np.newaxis, :], \
                   np.array(enc_cat)[np.newaxis, :], \
                   np.array(enc_time)[np.newaxis, :], \
                   np.array(enc_windows), \
                   np.array([real_lenth]), \
                   np.array(poi_category), \
                   np.array([label]),\
                    np.array(poi_candidate),\
                    is_real,
        else:
            return None,None,None,None,None,None,None,None,None,None,None

    def week_number(self,weekday):
        if weekday<=4:
            weekday=0
        else:
            weekday = 12
        return weekday
