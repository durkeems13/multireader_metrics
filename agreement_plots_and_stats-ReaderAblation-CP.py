#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:56:59 2023

@author: durkeems
"""
import warnings
warnings.simplefilter('ignore')
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress,mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sn
import pingouin as pg

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

# directories
rootdir = '/'.join(os.getcwd().split('/')[:-1])
aggdir = 'data/agreement_matrices'
matchdir = 'data/match_matrices'
save_dir = 'results/metrics'
plot_dir = 'results/plots'

adir = os.path.join(rootdir,aggdir)
amats = os.listdir(adir)
mdir = os.path.join(rootdir,matchdir)
comps = os.listdir(mdir)
sdir = os.path.join(rootdir,save_dir)
pdir = os.path.join(rootdir,plot_dir)
if not os.path.exists(sdir):
    os.makedirs(sdir)
if not os.path.exists(pdir):
    os.makedirs(pdir)
    
comps = [x for x in comps if '+CellPose' in x]
comps.append('HumansOnly')
    
if not os.path.exists(os.path.join(sdir,'master_ablation-CP_df.csv')):
    master_df = pd.DataFrame()
    for comp in comps:
        mr_dfs = os.listdir(os.path.join(mdir,comp))
        if comp=='Humans+CellPose-NoE':
            amats2 = [x for x in amats if 'NoE' in x]
            amats2 = [x for x in amats2 if 'CellPose' in x]
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        elif comp=='Humans+CellPose-NoD':
            amats2 = [x for x in amats if 'NoD' in x]
            amats2 = [x for x in amats2 if 'CellPose' in x]
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        elif comp=='Humans+CellPose-NoC':
            amats2 = [x for x in amats if 'NoC' in x]
            amats2 = [x for x in amats2 if 'CellPose' in x] 
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        elif comp=='Humans+CellPose-NoB':
            amats2 = [x for x in amats if 'NoB' in x]
            amats2 = [x for x in amats2 if 'CellPose' in x]
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        elif comp=='Humans+CellPose-NoA':
            amats2 = [x for x in amats if 'NoA' in x]
            amats2 = [x for x in amats2 if 'CellPose' in x]
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        elif comp=='HumansOnly':
            amats2 = [x for x in amats if '-No' not in x]
            amats2 = [x for x in amats2 if 'CellPose' not in x]
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        elif comp=='Humans+CellPose':
            amats2 = [x for x in amats if '-No' not in x]
            amats2 = [x for x in amats2 if 'CellPose' in x]
            amats2 = [x for x in amats2 if 'Bbox' not in x]
        
        compname=os.path.join(sdir,comp+'.csv')
        if os.path.exists(compname):
            comp_df = pd.read_csv(compname)
        else:
            comp_df = pd.DataFrame()
            imList = os.listdir(os.path.join(mdir,comp))
            imnames = ['_'.join(x.split('_')[:-2]) for x in imList]
            thList = [0.25,0.5,0.75]
            for th in thList:
                for im in imnames:
                    PWmats = [x for x in amats2 if im in x]
                    PWJI=[]
                    PWSI=[]
                    for PWmat in PWmats:
                        a = np.load(os.path.join(adir,PWmat),allow_pickle=True).item()
                        #print(a.keys())
                        am = a['AgreementMatrix']
                        R1ct,R2ct = np.shape(am)
                        am = np.where(am>th,am,0)
                        TP = np.sum(np.count_nonzero(am))
                        FN = R1ct-TP
                        FP = R2ct-TP
                        if (TP+FP+FN)>0:
                            JI = TP/(TP+FP+FN)
                            SI = 2*TP/(2*TP+FP+FN)
                        else:
                            JI = 1
                            SI = 1
                        PWJI.append(JI)
                        PWSI.append(SI)
                    MPWJI = np.mean(PWJI)
                    MPWSI = np.mean(PWSI)
                    
                    mr_df = [x for x in mr_dfs if (im in x) and (str(th) in x)]
                    print(mr_df)
                    MR_df = pd.read_csv(os.path.join(mdir,comp,mr_df[0]))
                    cols = MR_df.columns
                    cols = [x for x in cols if not 'Unnamed' in x]
                    MR_df2 = MR_df[cols]
                    obj,Nr = MR_df2.shape
                    preds = MR_df2.count().sum()
                    
                    MR_JI = (preds-obj)/((Nr-1)*obj)
                    MR_SI = Nr*(preds-obj)/(((Nr-1)**2)*obj+preds-obj) #Nr*MR_JI/(Nr-1+MR_JI)
                    
                    imdict = {'ImageName':[im],'Comparison':[comp],'IOUth':[th],'NumReaders':[Nr],'MPW_JI':[MPWJI],'MPW_SI':[MPWSI],'MR_JI':[MR_JI],'MR_SI':[MR_SI]}
                    imdf = pd.DataFrame.from_dict(imdict)
                    comp_df = pd.concat([comp_df,imdf],ignore_index=True)
        master_df = pd.concat([master_df,comp_df],ignore_index=True)
        if not os.path.exists(compname):
            comp_df.to_csv(compname)  
    master_df.to_csv(os.path.join(sdir,'master_ablation-CP_df.csv'))
else:
    master_df = pd.read_csv(os.path.join(sdir,'master_ablation-CP_df.csv'))
    master_df = master_df[master_df['Comparison']!='HumansOnly-Bbox']

#plot agreement against IOU
fig = plt.figure(figsize=(6,3),dpi=600)
sn.violinplot(data=master_df,x='Comparison',y='MR_JI',hue='IOUth',
              order=['Humans+CellPose','Humans+CellPose-NoE','Humans+CellPose-NoA',
                     'Humans+CellPose-NoB','Humans+CellPose-NoC','Humans+CellPose-NoD',
                     'HumansOnly'])
plt.xlabel('')
plt.ylabel('Multi-reader Jaccard Index')
#plt.xticks(ticks=[0,1,2,3,4,5,6],rotation=90)
plt.xticks(ticks=[0,1,2,3,4,5,6],rotation=0,labels=['All+CP','Excl. R1','Excl. R2',
                                                   'Excl. R3','Excl. R4',
                                                   'Excl. R5','Excl. CP'])
plt.ylim([0,1])
plt.legend(bbox_to_anchor=(1.17,1.05))
plt.tight_layout()
plt.savefig(os.path.join(pdir,'HumansOnly_JI_ReaderAblation-CP_MR.png'))

#plot agreement against IOU
fig = plt.figure(figsize=(6,3),dpi=600)
sn.violinplot(data=master_df,x='Comparison',y='MR_SI',hue='IOUth',
              order=['Humans+CellPose','Humans+CellPose-NoE','Humans+CellPose-NoA',
                     'Humans+CellPose-NoB','Humans+CellPose-NoC','Humans+CellPose-NoD',
                     'HumansOnly'])
plt.xlabel('')
plt.ylabel('Multi-reader Sorensen Index')
#plt.xticks(ticks=[0,1,2,3,4,5,6],rotation=90)
plt.xticks(ticks=[0,1,2,3,4,5,6],rotation=0,labels=['All+CP','Excl. R1','Excl. R2',
                                                   'Excl. R3','Excl. R4',
                                                   'Excl. R5','Excl. CP'])
plt.ylim([0,1])
plt.legend(bbox_to_anchor=(1.0,1.05))
plt.tight_layout()
plt.savefig(os.path.join(pdir,'HumansOnly_SI_ReaderAblation-CP_MR.png'))

JIpvals=[]
SIpvals=[]
compnames = []
for comp in comps:
    if '-No' not in comp:
        if 'Only' not in comp:
            continue
    for th in [0.25,0.5,0.75]:
        cname = 'HumansOnly vs '+comp+' at '+str(th)
        compnames.append(cname)
        sdf = master_df[master_df['IOUth']==th]
        s1 = sdf[sdf['Comparison']=='Humans+CellPose']
        s2 = sdf[sdf['Comparison']==comp]
        ustat,pval = mannwhitneyu(s2['MR_JI'],s1['MR_JI'],alternative='greater')
        JIpvals.append(pval)
        ustat,pval = mannwhitneyu(s2['MR_SI'],s1['MR_SI'],alternative='greater')
        SIpvals.append(pval)
        
JIpvals = [x*len(JIpvals)*3 for x in JIpvals]
SIpvals = [x*len(SIpvals)*3 for x in SIpvals]

for th in [0.25,0.5,0.75]:
    [print(x,'Jaccard pval:',JIpvals[i]) for i,x in enumerate(compnames) if str(th) in x]
    [print(x,'Sorensen pval:',SIpvals[i]) for i,x in enumerate(compnames) if str(th) in x]
        
#plot by sample (show only 0.25)
imnames = list(master_df['ImageName'])
snames = ['_'.join(x.split('_')[0]) for x in imnames]
usnames = np.unique(snames)
s_anon = {usnames[0]:'S5',usnames[1]:'S4',usnames[2]:'S3',
          usnames[3]:'S1',usnames[4]:'S2'}
anon_names = [s_anon[x] for x in snames]
master_df['SampleName']=anon_names

sub_df = master_df[master_df['IOUth']==0.25]
fig = plt.figure(figsize=(6,3),dpi=600)
sn.swarmplot(data=sub_df,x='Comparison',y='MR_JI',hue='SampleName',
              order=['Humans+CellPose','Humans+CellPose-NoE','Humans+CellPose-NoA',
                     'Humans+CellPose-NoB','Humans+CellPose-NoC','Humans+CellPose-NoD',
                     'HumansOnly'])
plt.xlabel('')
#plt.xticks(ticks=[0,1,2,3,4,5,6],rotation=90)
plt.xticks(ticks=[0,1,2,3,4,5,6],rotation=0,labels=['All+CP','Excl. R1','Excl. R2',
                                                   'Excl. R3','Excl. R4',
                                                   'Excl. R5','Excl. CP'])
plt.ylabel('Multi-reader Jaccard Index')
plt.ylim([0,1])
plt.legend(bbox_to_anchor=(1.0,1.05))
plt.tight_layout()
plt.savefig(os.path.join(pdir,'HumansOnly_JI_ReaderAblation-CP_bySample.png'))






