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
mat_cr=np.zeros((33,14), dtype='float32')
mat_ct=np.zeros((33,14), dtype='float32')
mat_dt=np.zeros((33,14), dtype='float32')

def read_pFPC(sizes):
    df=pd.read_csv('/tmp/pfpc_res.txt',header=None)
    pfpc=df.values
    pfpc=pfpc.reshape(-1,3)
    ctm=pfpc[:,1]
    dtm=pfpc[:,2]
    cr=pfpc[:,0] 
    ct=sizes*1000/(ctm*1024**3)
    dt=sizes*1000/(dtm*1024**3)
    return (cr,ct,dt)

def read_SPDP(sizes):
    df=pd.read_csv('/tmp/spdp_res.txt',header=None)
    spdp=df.values
    spdp=spdp.reshape(-1,3)
    ctm=spdp[:,1]
    dtm=spdp[:,2]
    cr=spdp[:,0] 
    ct=sizes*1000/(ctm*1024**3)
    dt=sizes*1000/(dtm*1024**3)
    return (cr,ct,dt)

def read_fpzip(sizes):
    df=pd.read_csv('/tmp/fpzip_res.txt',header=None, sep=' ')
    fpzip=df.values
    fpzip[8]=([0,0,0])
    fpzip=fpzip.astype('float32')
    ctm=fpzip[:,1]
    dtm=fpzip[:,2]
    cr=fpzip[:,0] 
    ct=sizes*1000/(ctm*1024**3)
    dt=sizes*1000/(dtm*1024**3)
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
    return (cr,ct,dt)                

def read_ndzc(sizes):
    df=pd.read_csv('/tmp/ndzc_res.txt',header=None, sep=' ')
    ndzc=df.values
    ndzc[15]=[0,0]
    ndzc1=ndzc[:33].astype('float32')
    ndzc2=ndzc[33:].astype('float32')
    cr=ndzc1[:,0] 
    ctm=ndzc1[:,1]
    dtm=ndzc2[:,0]
    ct=sizes/(ctm*1024**3)
    dt=sizes/(dtm*1024**3)
    return (cr,ct,dt)

def read_ndzg(sizes):
    df=pd.read_csv('/tmp/ndzg_res.txt',header=None, sep=' ')
    ndzg=df.values
    ndzg[14]=[0,0]
    ndzg1=ndzg[:32].astype('float32')
    ndzg2=ndzg[32:].astype('float32')
    ndzg1=np.insert(ndzg1,6,[0,0],axis=0)
    ndzg2=np.insert(ndzg2,6,[0,0],axis=0)
    cr=ndzg1[:,0] 
    ctm=ndzg1[:,1]
    dtm=ndzg2[:,0]
    ct=sizes/(ctm*1024**3)
    dt=sizes/(dtm*1024**3)
    return (cr,ct,dt)

def main(work):
    if work == 'cr':
        print("         compression ratio (orig-szie/comp-size)")
    if work == 'ct':
        print("         compression speed (GB/s)")
    if work == 'dt':
        print("         decompression speed (GB/s)")
    print("==================================================="*2)
    print("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format
         (" ","pfpc","spdp","fpzip","shf+LZ4","shf+zstd","ndzip-CPU","BUFF",
          "gorilla","chimp","GFC","MPC","nv:LZ4","nv:bitcomp","ndzip-GPU"))
    for i in range(33):
    print("---------------------------------------------------"*2)

if __name__ == "__main__":
    main(work)
