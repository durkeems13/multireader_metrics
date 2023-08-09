#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:47:30 2023

@author: durkeems
"""
import warnings
warnings.simplefilter('ignore')
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import seaborn as sn

#SET PLOTTING PARAMETERS
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE, family='sans', variant='small-caps')         # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',type=str,default='/'.join(os.getcwd().split('/')[:-1]),help='')
parser.add_argument('--mmat_dir',type=str,default='data/match_matrices/Humans+CellPose',help='directory to save output to')
parser.add_argument('--consensus_dir',type=str,default='results/consensus',help='directory to save output to')

args,unparsed=parser.parse_known_args()

mmatdir = os.path.join(args.root_dir,args.mmat_dir)
consensusdir = os.path.join(args.root_dir,args.consensus_dir)

if not os.path.exists(consensusdir):
    os.makedirs(consensusdir)
human_readers = ['ReaderA','ReaderB','ReaderC','ReaderD','ReaderE']
 
matchmats = os.listdir(mmatdir)
matchmats = [x for x in matchmats if x.endswith('0.5.csv')]

Rn = len(human_readers)
master_df = pd.DataFrame()
for i in np.arange(1,Rn+1):
    JI = []
    SI = []
    ims = []
    for j,m in enumerate(matchmats):
        df = pd.read_csv(os.path.join(mmatdir,m))
        dfC = df.copy()
        rcols = [x for x in df.columns if 'Reader' in x]
        dfR = df[rcols]
        dfR['CallCount']=Rn-dfR.isnull().sum(axis=1)
        dfC['CallCount']=Rn-dfR.isnull().sum(axis=1)
        dfR2 = dfR[dfR['CallCount']>=i]
        dfC2 = dfC[dfC['CallCount']>=i]
        
        allcalls,_=df.shape
        obj,_ = dfR2.shape

        cpMisses = df['CellPose'].isnull().sum()
        cpCalls = allcalls-cpMisses

        FN = dfC2['CellPose'].isnull().sum()
        
        TP = obj-FN
        FP = cpCalls-TP

        JI.append(TP/(TP+FP+FN))
        SI.append(2*TP/(2*TP+FP+FN))
        
        ims.append(m.split('_TH')[0])
        
    #plot humans only vs humans + CP        
    I = [i]*len(ims)
    imlist = [{'ImageName':x,'JI':float(y),'SI':float(z),
                'NumCallsGT':int(k)} for x,y,z,k in zip(ims,JI,SI,I)]

    data = pd.DataFrame.from_records(imlist)
    master_df = pd.concat([master_df,data],ignore_index=True)  
                 
fig = plt.figure(figsize=(6,3),dpi=600)
sn.violinplot(data=master_df,x='NumCallsGT',y='JI')#,hue='NumCallsGT')
plt.xticks([0,1,2,3,4],labels=['1','2','3','4','5'])
plt.xlabel('# Calls per GT cell')
plt.ylabel('Jaccard Index')
#plt.legend(loc='lower right')
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(consensusdir,'JaccardIndex_HconsensusvHCP.png'))
plt.show()

fig = plt.figure(figsize=(6,3),dpi=600)
sn.violinplot(data=master_df,x='NumCallsGT',y='SI')#,hue='NumCallsGT')
plt.xticks([0,1,2,3,4],labels=['1','2','3','4','5'])
plt.xlabel('# Calls per GT cell')
plt.ylabel('Sorensen Index')
#plt.legend(loc='lower right')
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(consensusdir,'SorensenIndex_HconsensusvHCP.png'))
plt.show()
        
s1 = master_df[master_df['NumCallsGT']==1]['JI']
s2 = master_df[master_df['NumCallsGT']==2]['JI']
s3 = master_df[master_df['NumCallsGT']==3]['JI']
s4 = master_df[master_df['NumCallsGT']==4]['JI']
s5 = master_df[master_df['NumCallsGT']==5]['JI']

S,P = kruskal(s1,s2,s3,s4,s5)
print(P)

s1 = master_df[master_df['NumCallsGT']==1]['SI']
s2 = master_df[master_df['NumCallsGT']==2]['SI']
s3 = master_df[master_df['NumCallsGT']==3]['SI']
s4 = master_df[master_df['NumCallsGT']==4]['SI']
s5 = master_df[master_df['NumCallsGT']==5]['SI']

S,P = kruskal(s1,s2,s3,s4,s5)
print(P)
