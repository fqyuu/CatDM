from collections import defaultdict
from math import radians, cos, sin, asin, sqrt


def filter_to_generate(user,topk_cat,current_poi,read_path_venue,write_path):
    poi_dict = defaultdict(list)

    user_behavior=defaultdict(list)

    with open(read_path_venue) as f1:
        for line in f1:
            i, vc, lat, lon= line.strip().split(",")
            i, vc = map(int, [i, vc])
            lat, lon=map(float,[lat, lon])
            # print(u)
            poi_dict[i-1].append([i-1,vc - 1,lat,lon])
    with open(write_path, 'a') as f:
        count_before=0
        count_after=0
        for key in poi_dict:
            if poi_dict[key][0][1] in topk_cat:
                latitude=poi_dict[key][0][2]
                longitude=poi_dict[key][0][3]
                count_before +=1
                if haversine(longitude,latitude,poi_dict[current_poi][0][3],poi_dict[current_poi][0][2])<8:
                   f.write("{:},{:},{:}\n".format(user, key, poi_dict[key][0][1]))
                   count_after += 1
        # print("user:{:},beforefilter:{:},afterfilter:{:}".format(user, count_before,count_after))

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r
