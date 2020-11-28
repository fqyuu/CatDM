import numpy as np


nyc_filepath="../../data/Foursquare/NYC/NYC_PROCESS_DELETE.csv"
nyc_minut_count=[]
for i in range(1440):
    nyc_minut_count.append(0)

with open(nyc_filepath) as f1:
    for line in f1:
        _,_,_,_,_,time=line.strip().split(",")
        nyc_minut_count[int(time[-8:-6])*60+int(time[-5:-3])]+=1



nyc_sum=np.sum(nyc_minut_count)
nyc_minut_pro=np.divide(nyc_minut_count,nyc_sum)

record_time_nyc=[]
probability_nyc=float(0)
record_time_nyc.append(0)
for i,line in enumerate(nyc_minut_pro):
    probability_nyc+=line
    if(probability_nyc>=(1/12)):
        print("i:{}     probability:{}".format(i,probability_nyc))
        probability_nyc = float(0)
        record_time_nyc.append(i)
record_time_nyc.append(1439)



