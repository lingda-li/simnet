import sys
import os
import numpy as np
import time
import pandas as pd
import ipdb
#datasets= ['557.xz_r']
datasets= ['554.roms_r','997.specrand_fr', '507.cactuBSSN_r', '531.deepsjeng_r', '538.imagick_r',
        '505.mcf_r',
        '519.lbm_r',
        '521.wrf_r',
        '523.xalancbmk_r',
        '525.x264_r',
        '526.blender_r',
        '527.cam4_r',
        '544.nab_r',
        '548.exchange2_r',
        '549.fotonik3d_r',
        '999.specrand_ir',
        '557.xz_r']
#datasets= pd.Series(datasets)
r_1M= [
1151709,
1630812,
1151709,
1151709,
1147025,
1147025,
1151709,
1151709,
1151709,
1147025,
1151709,
1151709,
1147025,
1630812,
1151709,
1151709,
1151709
]


r_10M= [
11858863,
16816825,
13896084,
12487783,
13661432,
11366922,
13603833,
13993858,
13451087,
14203839,
13851521,
13847872,
14321243,
11026001,
14488298,
16298570,
13515446
]


r_100M= [
126417300,
160600342,
133347088,
140394175,
145651271,
116977034,
143198169,
172726488,
139071680,
148966885,
152308260,
165801643,
148723255,
115789665,
127189455,
160872998,
145067659     
        ]

custom_list= ['527.cam4_r','544.nab_r','548.exchange2_r','549.fotonik3d_r']
custom_list= ['523.xalancbmk_r', '525.x264_r', '526.blender_r', '527.cam4_r',
               '544.nab_r', '548.exchange2_r', '549.fotonik3d_r',
                      '999.specrand_ir']
def check(predicted,dataset,warmup):
    index= datasets.index(dataset)
    #index= Index(datasets).get_loc(dataset) 
    truth= r_10M[index]
    error= (abs(truth-predicted)/truth)*100
    #print(dataset, predicted, truth, error,w)
    #print(dataset, error)
    return error

fileName= sys.argv[1]
'''
CNN3_2048_half.engine,505.mcf_r.qq100m.tra.bin,121261539,4491.97,2048,1,121261539,44.9197
'''
Colname= ['Executable', 'model', 'dataset', 'instructions', 'subtraces', 'warmup', 'max iteration', 'prediction', 'time']
#Colname= ['model', 'dataset', 'prediction1', 'time1', 'subtraces', 'GPUs', 'prediction', 'time']
df= pd.read_csv(fileName, names=Colname, header=None)	 
df['prediction']=df['prediction'].astype(int)
df['time']=df['time'].astype(float)
df['dataset']=df['dataset'].str.replace(".qq100m.tra.bin","")
#df=df[df.dataset.isin(custom_list)]
#ipdb.set_trace()
df= df[df['subtraces']==131072]
#ipdb.set_trace()
df= df[df['warmup']==0]

inter= ['507.cactuBSSN_r', '527.cam4_r', '997.specrand_fr', '557.xz_r', '523.xalancbmk_r', '999.specrand_ir', '554.roms_r', '526.blender_r', '519.lbm_r', '505.mcf_r', '549.fotonik3d_r', '525.x264_r', '548.exchange2_r', '538.imagick_r', '544.nab_r']

df=df[df.dataset.isin(inter)]


error= []
for i in range(df.shape[0]):
    pr= df['prediction'].iloc[i]
    dt= df['dataset'].iloc[i]
    w= df['subtraces'].iloc[i]
    err= check(pr,dt,w)
    error.append(err)
#df['Error']= df.apply(lambda x:check(df['prediction'],df['dataset']),axis=1)
df['error']= error
#print(df)
#df_g= df.groupby(['threshold','warmup','subtraces'])['error','time'].mean()
#df_g=df_g.sort_values(['threshold','warmup'],ascending=True)
df_g= df.groupby(['dataset'])['error'].mean()
#df_g= df.groupby(['subtraces','warmup'])['error','time'].mean()

#df_gg=df.groupby(['subtraces'])['dataset'].count()
#df_g= df.groupby(['subtraces','threshold'])['error','time'].mean()
print(df_g)
#print(df_gg)
#exit()
warmup= [0,2,4,8,16,32,64,96,128,160,192,200]
warmup= [1,2,4,8,200]
threshold= [2, 4, 8, 10, 16, 20, 30, 40,100,200,300]
#warmup= [3,5,6,7]
#df_formatted= pd.DataFrame()
xx=df_g.reset_index()
ipdb.set_trace()
print(xx)
xx.to_csv('scatter_graph.log',index=False)
for t in threshold:
    df_formatted= pd.DataFrame()
    for w in warmup:
        #print(t)
        #ipdb.set_trace()
        #df_formatted[str(w)]= xx['error'][xx['warmup']==w].values
        zy= xx[['error','threshold']][xx['warmup']==w]
        df_formatted[str(w)]=zy['error'][zy['threshold']==t].values
                #print(df_formatted)
    print("Threshold: ", t)
    print(df_formatted)
    
    #ipdb.set_trace()
print("***********************")
df_gg= df.groupby(['warmup','subtraces','threshold'])['time'].mean()
df_formatted_t= pd.DataFrame()
yy= df_gg.reset_index()
#ipdb.set_trace()

for t in threshold:
    df_formatted= pd.DataFrame()
    for w in warmup:
        #print(t)
        #ipdb.set_trace()
        #df_formatted[str(w)]= xx['error'][xx['warmup']==w].values
        zy= yy[['time','threshold']][yy['warmup']==w]
        df_formatted[str(w)]=zy['time'][zy['threshold']==t].values
                #print(df_formatted)
    print("Threshold: ", t)
    print(df_formatted)

#print(df_formatted)
#df_g['time']= df_gg
#print(df_formatted_t)
df_formatted.round(2)
df_formatted_t.round(2)
#ipdb.set_trace()
