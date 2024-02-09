# coding: utf-8
import numpy as np
import pandas as pd
import os
import time
import subprocess as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--work', type=str, default="cr")  # ct, dt

args = parser.parse_args()
work = args.work

algs = ['pfpc', 'spdp', 'fpzp', 'shlz', 'shzd', 'ndzc', 'buff', 'grla', 'chmp', 'gfc', 'mpc', 'nvlz', 'nvbm', 'ndzg']
orig_size={
    'msg-bt': 266389432, 'num-brain':141840000, 'num-control':159504744, 'rsim':94281728, 'astro-mhd':548458560,
    'astro-pt':671088640, 'miranda3d': 4294967296, 'turbulance': 67108864, 'wave':536870912, 'hurricane':100000000,
    'citytemp':11625304, 'ts-gas':307452800, 'phone-gyro':334383168, 'wesad-chest': 272339200, 'jane-street':1810997760,
    'nyc-taxi':713711376, 'gas-price':886619664, 'solar-wind':423980536, 'acs-wht':225000000, 'hdr-night':536870912,
    'hdr-palermo':843454592, 'hst-wfc3-uvis':108924760, 'hst-wfc3-ir':24015312, 'spitzer-irac':164989536, 'g24-78-usb':1335668264,
    'jws-mirimage':169082880,'tpcH-order':120000000, 'tpcxBB-store':789920928, 'tpcxBB-web':986782680, 'tpcH-lineitem':959776816,
    'tpcDS-catalog':172803480, 'tpcDS-store':276515952, 'tpcDS-web': 86354820 
}
sizes = np.array(list(orig_size.values()))
datas = np.array(list(orig_size.keys()))
mat_cr=np.zeros((33,14), dtype='float32')
mat_ct=np.zeros((33,14), dtype='float32')
mat_dt=np.zeros((33,14), dtype='float32')

def read_pFPC(sizes):
    df=pd.read_csv('/tmp/pfpc_res.txt',header=None)
    pfpc=df.values
    pfpc=pfpc.reshape(3,-1)
    ctm=pfpc[1]
    dtm=pfpc[2]
    cr=pfpc[0] 
    ct=sizes*1000/(ctm*1024**3)
    dt=sizes*1000/(dtm*1024**3)
    return (cr,ct,dt)

def read_SPDP(sizes):
    df=pd.read_csv('/tmp/spdp_res.txt',header=None)
    spdp=df.values
    spdp=spdp.reshape(3,-1)
    ctm=spdp[1]
    dtm=spdp[2]
    cr=spdp[0] 
    ct=sizes*1000/(ctm*1024**3)
    dt=sizes*1000/(dtm*1024**3)
    return (cr,ct,dt)

def read_fpzip(sizes):
    df=pd.read_csv('/tmp/fpzip_res.txt',header=None, sep=' ')
    fpzip=df.values
    fpzip[8]=([0,1e20,1e20])  # wave
    fpzip=fpzip.astype('float32')
    fpzip=np.insert(fpzip,9,[0,1e20,1e20],axis=0) # hurrican
    ctm=fpzip[:,1]
    dtm=fpzip[:,2]
    cr=fpzip[:,0] 
    ct=sizes*1000/(ctm*1024**3) # time in ms
    ct[8] = 0.0
    ct[9] = 0.0
    dt=sizes*1000/(dtm*1024**3) # time in ms
    dt[8] = 0.0
    dt[9] = 0.0
    return (cr,ct,dt) 

def read_shflz(sizes):
    df=pd.read_csv('/tmp/shflz_res.txt',header=None, sep=' ')
    shflz=df.values
    shflz=shflz.reshape(-1,4)[:,:3]
    shflz[6]=([0,0,0])
    shflz=shflz.astype('float32')
    cr=shflz[:,1] 
    ct=shflz[:,0]
    dt=shflz[:,2]
    cr = np.insert(cr, 6, 0) # miranda
    ct = np.insert(ct, 6, 0) # miranda
    dt = np.insert(dt, 6, 0) # miranda
    cr = np.insert(cr, 6, 32) # tpcds-web
    ct = np.insert(ct, 6, 32) # tpcds-web
    dt = np.insert(dt, 6, 32) # tpcds-web
    return (cr,ct,dt)

def read_shfzstd(sizes):
    df=pd.read_csv('/tmp/shflz_res.txt',header=None, sep=' ')
    shfzstd=df.values
    shfzstd=shfzstd.reshape(-1,4)[:,:3]
    shfzstd[6]=([0,0,0])
    shfzstd=shfzstd.astype('float32')
    cr=shfzstd[:,1] 
    ct=shfzstd[:,0]
    dt=shfzstd[:,2]
    cr = np.insert(cr, 6, 0) # miranda
    ct = np.insert(ct, 6, 0) # miranda
    dt = np.insert(dt, 6, 0) # miranda
    cr = np.insert(cr, 6, 32) # tpcds-web
    ct = np.insert(ct, 6, 32) # tpcds-web
    dt = np.insert(dt, 6, 32) # tpcds-web
    return (cr,ct,dt)                

def read_ndzc(sizes):
    df=pd.read_csv('/tmp/ndzc_res.txt',header=None, sep=' ')
    ndzc=df.values.reshape(3,-1)
    cr=ndzc[0] 
    ctm=ndzc[1]
    dtm=ndzc[2]
    ctm[15]=1e20
    dtm[15]=1e20
    ct=sizes*1000/(ctm*1024**3) # time in ms
    dt=sizes*1000/(dtm*1024**3) # time in ms
    ct[15]=0 # nyctaxi 
    dt[15]=0 # nyctaxi 
    return (cr,ct,dt)

def read_ndzg(sizes):
    df=pd.read_csv('/tmp/ndzg_res.txt',header=None, sep=' ')
    ndzg=df.values
    ndzg=ndzg.reshape(3,-1)
    cr=ndzg[0] 
    ctm=ndzg[1]
    dtm=ndzg[2]
    cr=np.insert(cr,6,0.0)
    ctm=np.insert(ctm,6,1e20)
    dtm=np.insert(dtm,6,1e20)
    ctm[15]=1e20
    dtm[15]=1e20
    ct=sizes*1000/(ctm*1024**3) # time in ms
    dt=sizes*1000/(dtm*1024**3) # time in ms
    ct[6]=0
    dt[6]=0
    ct[15]=0 # nyctaxi 
    dt[15]=0 # nyctaxi 
    return (cr,ct,dt)

def read_buff(sizes):
    df=pd.read_csv('/tmp/buff_res.txt',header=None, sep=',')
    buff=df.values
    cr=buff[:,0] 
    ct=buff[:,1] 
    dt=buff[:,2] 
    cr=np.insert(cr, 9, 0)
    ct=np.insert(ct, 9, 0)
    dt=np.insert(dt, 9, 0)
    return (cr,ct,dt)

def read_gorilla(sizes):
    df=pd.read_csv('/tmp/gorilla_res.txt',header=None, sep=' ')
    gorilla=df.values
    cr=gorilla[:,0] 
    ctm=gorilla[:,1]
    dtm=gorilla[:,0]
    ct=sizes/(ctm*1024**3) # time in s
    dt=sizes/(dtm*1024**3) # time in s
    return (cr,ct,dt)

def read_chimp(sizes):
    df=pd.read_csv('/tmp/chimp_res.txt',header=None, sep=' ')
    chimp=df.values
    cr=chimp[:,0] 
    ctm=chimp[:,1]
    dtm=chimp[:,0]
    ct=sizes/(ctm*1024**3) # time in s
    dt=sizes/(dtm*1024**3) # time in s
    return (cr,ct,dt)

def read_gfc(sizes):
    df=pd.read_csv('/tmp/gfc_res.txt',header=None, sep=' ')
    gfc=df.values
    gfc=gfc.reshape(3,-1)
    cr=gfc[0] 
    ct=gfc[1] 
    dt=gfc[2]
    # cr=np.insert(cr,[4,5,6,14,15,16,20,24,27,28,29],[0,0,0,0,0,0,0,0,0,0,0]) 
    cr=np.insert(cr,[4,4,4],[0,0,0])
    cr=np.insert(cr,[14,14,14],[0,0,0])
    cr=np.insert(cr,20,0)
    cr=np.insert(cr,24,0)
    cr=np.insert(cr,[27,27,27],[0,0,0])
    ct=np.insert(ct,[4,4,4],[0,0,0])
    ct=np.insert(ct,[14,14,14],[0,0,0])
    ct=np.insert(ct,20,0)
    ct=np.insert(ct,24,0)
    ct=np.insert(ct,[27,27,27],[0,0,0])
    dt=np.insert(dt,[4,4,4],[0,0,0])
    dt=np.insert(dt,[14,14,14],[0,0,0])
    dt=np.insert(dt,20,0)
    dt=np.insert(dt,24,0)
    dt=np.insert(dt,[27,27,27],[0,0,0])
    return (cr,ct,dt)

def read_mpc(sizes):
    df=pd.read_csv('/tmp/mpc_res.txt',header=None, sep=' ')
    mpc=df.values
    mpc=mpc.reshape(3,-1)
    cr=mpc[1] 
    ct=mpc[0] 
    dt=mpc[2] 
    return (cr,ct,dt)

def read_nvlz4(sizes):
    df=pd.read_csv('/tmp/nvlz4_res.txt',header=None, sep=' ')
    nvlz4=df.values
    nvlz4=nvlz4.reshape(3,-1)
    cr=nvlz4[0] 
    ct=nvlz4[1] 
    dt=nvlz4[2] 
    return (cr,ct,dt)

def read_nvbitcomp(sizes):
    df=pd.read_csv('/tmp/nvbitcomp_res.txt',header=None, sep=' ')
    nvbitcomp=df.values
    nvbitcomp=nvbitcomp.reshape(3,-1)
    cr=nvbitcomp[0] 
    ct=nvbitcomp[1] 
    dt=nvbitcomp[2] 
    return (cr,ct,dt)

def main(work):
    if work == 'cr':
        print("         compression ratio (orig-szie/comp-size)")
    if work == 'ct':
        print("         compression speed (GB/s)")
    if work == 'dt':
        print("         decompression speed (GB/s)")
    pfpc=read_pFPC(sizes)
    spdp=read_SPDP(sizes)
    fpzip=read_fpzip(sizes)
    shflz=read_shflz(sizes)
    shfzstd=read_shfzstd(sizes)
    ndzc=read_ndzc(sizes)
    # buff=read_buff(sizes)
    gorilla=read_gorilla(sizes)
    chimp=read_chimp(sizes)
    gfc=read_gfc(sizes)
    mpc=read_mpc(sizes)
    nvlz4=read_nvlz4(sizes)
    nvbitcomp=read_nvbitcomp(sizes)
    ndzg=read_ndzg(sizes)
    print("========================================================="*2)
    # print("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format
    #      (" ","pfpc","spdp","fpzip","shf+LZ4","shf+zstd","ndzip-CPU","BUFF",
    #       "gorilla","chimp","GFC","MPC","nv:LZ4","nv:bitcomp","ndzip-GPU"))
    print(f"{' ':>13} {algs[0]:>6} {algs[1]:>6} {algs[2]:>6} {algs[3]:>6} {algs[4]:>6} {algs[5]:>6} {algs[7]:>6} {algs[8]:>6} {algs[9]:>6} {algs[10]:>6} {algs[11]:>6} {algs[12]:>6} {algs[13]:>6}")
    if work == 'cr':
        for i in range(33):
            print(f"{datas[i]:>13} {pfpc[0][i]:>6.2f} {spdp[0][i]:>6.2f} {fpzip[0][i]:>6.2f} \
                {shflz[0][i]:>6.2f} {shfzstd[0][i]:>6.2f} {ndzc[0][i]:>6.2f} {gorilla[0][i]:>6.2f} \
                {chimp[0][i]:>6.2f} {gfc[0][i]:>6.2f} {mpc[0][i]:>6.2f} {nvlz4[0][i]:>6.2f} {nvbitcomp[0][i]:>6.2f} {ndzg[0][i]:>6.2f}")
    if work == 'ct':
        for i in range(33):
            print(f"{datas[i]:>13} {pfpc[1][i]:>6.2f} {spdp[1][i]:>6.2f} {fpzip[1][i]:>6.2f} \
                {shflz[1][i]:>6.2f} {shfzstd[1][i]:>6.2f} {ndzc[1][i]:>6.2f} {gorilla[1][i]:>6.2f} \
                {chimp[1][i]:>6.2f} {gfc[1][i]:>6.2f} {mpc[1][i]:>6.2f} {nvlz4[1][i]:>6.2f} {nvbitcomp[1][i]:>6.2f} {ndzg[1][i]:>6.2f}")        
    if work == 'dt':
        for i in range(33):
            print(f"{datas[i]:>13} {pfpc[2][i]:>6.2f} {spdp[2][i]:>6.2f} {fpzip[2][i]:>6.2f} \
                {shflz[2][i]:>6.2f} {shfzstd[2][i]:>6.2f} {ndzc[2][i]:>6.2f} {gorilla[2][i]:>6.2f} \
                {chimp[2][i]:>6.2f} {gfc[2][i]:>6.2f} {mpc[2][i]:>6.2f} {nvlz4[2][i]:>6.2f} {nvbitcomp[2][i]:>6.2f} {ndzg[2][i]:>6.2f}")        
    # for i in range(33):
    #     print("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format
    #      (" ",pfpc[0][i],spdp[0][i],fpzip[0][i],shflz[0][i],shfzstd[0][i],ndzc[0][i],buff[0][i],
    #       gorilla[0][i],chimp[0][i],gfc[0][i],mpc[0][i],nvlz4[0][i],nvbitcomp[0][i],ndzg[0][i]))
    print("---------------------------------------------------"*2)

if __name__ == "__main__":
    main(work)
