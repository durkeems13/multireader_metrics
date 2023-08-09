#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:56:59 2023

@author: durkeems
"""
import warnings
warnings.simplefilter('ignore')
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress,mannwhitneyu,sem,t
import matplotlib.pyplot as plt
import seaborn as sn

def NI_test(X1,X2,r_diff,equal_var=False,increase_good=True):
    M1 = np.mean(X1)
    N1 = len(X1)
    S1 = np.std(X1)
    
    delta = r_diff*M1
    if increase_good:
        th = -1*delta
    else:
        th = delta
    
    M2 = np.mean(X2)
    N2 = len(X2)
    S2 = np.std(X2)
    
    diffs = [x-y for x,y in zip(X1,X2)]
    
    diffs.sort()
    dSE = sem(diffs)
    dH = dSE*t.ppf(0.9,49)
    LB = np.mean(diffs)-dH
    return LB,th

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
    
if not os.path.exists(os.path.join(sdir,'master_comparison_df.csv')):
    master_df = pd.DataFrame()
    for comp in comps:
        mr_dfs = os.listdir(os.path.join(mdir,comp))
        if comp=='HumansOnly':
            amats2 = [x for x in amats if 'Bbox' not in x]
            amats2 = [x for x in amats2 if 'Yolo' not in x]
            amats2 = [x for x in amats2 if 'CellPose' not in x]
            amats2 = [x for x in amats2 if '-No' not in x]
            amats2 = [x for x in amats2 if not (x.split('_')[-2]).endswith('2')]
            amats2 = [x for x in amats2 if not (x.split('_')[-4]).endswith('2')]
        elif comp=='HumansOnly-Bbox':
            amats2 = [x for x in amats if 'Bbox' in x]
            amats2 = [x for x in amats2 if 'Yolo' not in x]
            amats2 = [x for x in amats2 if 'CellPose' not in x]
            amats2 = [x for x in amats2 if '-No' not in x]
            amats2 = [x for x in amats2 if not (x.split('_')[-2]).endswith('2')]
            amats2 = [x for x in amats2 if not (x.split('_')[-4]).endswith('2')]
        elif comp=='Humans+Yolov5':
            amats2 = [x for x in amats if 'Bbox' in x]
            amats2 = [x for x in amats2 if 'CellPose' not in x]
            amats2 = [x for x in amats2 if '-No' not in x]
            amats2 = [x for x in amats2 if not (x.split('_')[-2]).endswith('2')]
            amats2 = [x for x in amats2 if not (x.split('_')[-4]).endswith('2')]
        elif comp=='Humans+CellPose':
            amats2 = [x for x in amats if 'Bbox' not in x]
            amats2 = [x for x in amats2 if 'Yolo' not in x]
            amats2 = [x for x in amats2 if '-No' not in x]
            amats2 = [x for x in amats2 if not (x.split('_')[-2]).endswith('2')]
            amats2 = [x for x in amats2 if not (x.split('_')[-4]).endswith('2')]
        elif comp=='HumanConsistency':
            amats2 = [x for x in amats if 'Bbox' not in x]
            amats2 = [x for x in amats2 if 'Yolo' not in x]
            amats2 = [x for x in amats2 if 'CellPose' not in x]
            amats2 = [x for x in amats2 if '-No' not in x]
        else:
            continue
        compname=os.path.join(sdir,comp+'.csv')
        comp_df = pd.DataFrame()
        imList = os.listdir(os.path.join(mdir,comp))
        imnames = ['_'.join(x.split('_')[:-2]) for x in imList]
        imnames = np.unique(imnames)
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
                MR_df = pd.read_csv(os.path.join(mdir,comp,mr_df[0]))
                cols = MR_df.columns
                cols = [x for x in cols if not 'Unnamed' in x]
                MR_df2 = MR_df[cols]
                obj,Nr = MR_df2.shape
                preds = MR_df2.count().sum()
                
                MR_JI = (preds-obj)/((Nr-1)*obj)
                MR_SI = Nr*(preds-obj)/(((Nr-1)**2)*obj+preds-obj) #Nr*MR_JI/(Nr-1+MR_JI)
                
                imdict = {'ImageName':[im],'Comparison':[comp],'IOUth':[th],'NumReaders':[Nr],'NumObj':[obj],'NumPreds':[preds],'MPW_JI':[MPWJI],'MPW_SI':[MPWSI],'MR_JI':[MR_JI],'MR_SI':[MR_SI]}
                imdf = pd.DataFrame.from_dict(imdict)
                comp_df = pd.concat([comp_df,imdf],ignore_index=True)
        master_df = pd.concat([master_df,comp_df],ignore_index=True)
        comp_df.to_csv(compname)
    master_df.to_csv(os.path.join(sdir,'master_comparison_df.csv'))
else:
    master_df = pd.read_csv(os.path.join(sdir,'master_comparison_df.csv'))

#sub dfs
HO_df = master_df[master_df['Comparison']=='HumansOnly']
HOB_df = master_df[master_df['Comparison']=='HumansOnly-Bbox']
HY_df = master_df[master_df['Comparison']=='Humans+Yolov5']
HCP_df = master_df[master_df['Comparison']=='Humans+CellPose']
HC_df = master_df[master_df['Comparison']=='HumanConsistency']
 

#plot agreement against IOU
fig = plt.figure(figsize=(4,3),dpi=600)
yerr1 = HO_df.groupby('IOUth')['MPW_JI'].std()
yerr2 = HO_df.groupby('IOUth')['MR_JI'].std()
HO_df.groupby('IOUth')['MPW_JI'].mean().plot(label='Human readers: Mean pairwise',yerr=yerr1)
HO_df.groupby('IOUth')['MR_JI'].mean().plot(label='Human readers: Multi-reader',yerr=yerr2)
plt.xlabel('IOU threshold')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_by_IOU_MPWvMR.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
yerr1 = HO_df.groupby('IOUth')['MPW_SI'].std()
yerr2 = HO_df.groupby('IOUth')['MR_SI'].std()
HO_df.groupby('IOUth')['MPW_SI'].mean().plot(label='Human readers: Mean pairwise',yerr=yerr1)
HO_df.groupby('IOUth')['MR_SI'].mean().plot(label='Human readers: Multi-reader',yerr=yerr2)
plt.xlabel('IOU threshold')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_by_IOU_MPWvMR.png'))
#plt.show()
plt.close()

#plot humans only PWM vs MR
d1 = HO_df[['ImageName','IOUth','MPW_SI','MPW_JI']]
d1['Metric']=0
d1=d1.rename(columns={'MPW_JI':'JI','MPW_SI':'SI'})
d2 = HO_df[['ImageName','IOUth','MR_SI','MR_JI']]
d2['Metric']=1
d2=d2.rename(columns={'MR_JI':'JI','MR_SI':'SI'})
data = pd.concat([d1,d2],ignore_index=True)

#sample swarm
fig = plt.figure(figsize=(4,3),dpi=600)
ims = list(HO_df['ImageName'])
samps = [x.split('_')[0] for x in ims]
usnames = np.unique(samps)
s_anon = {usnames[0]:'S5',usnames[1]:'S4',usnames[2]:'S3',
          usnames[3]:'S1',usnames[4]:'S2'}
anon = [s_anon[x] for x in samps]
HO_df['SampleName']=anon
sn.swarmplot(data=HO_df,x='IOUth',y='MR_JI',hue='SampleName')
plt.xticks([0,1,2],labels=['0.25','0.50','0.75'])
plt.ylabel('Jaccard Index')
plt.legend()
plt.xlabel('IOU threshold')
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_Swarm.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='Metric',y='JI',hue='IOUth')
plt.xticks([0,1],labels=['Mean pairwise','Multi-reader'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_MPWvMR.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='Metric',y='SI',hue='IOUth')
plt.xticks([0,1],labels=['Mean pairwise','Multi-reader'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_MPWvMR.png'))
#plt.show()
plt.close()

#plot humans only bbox vs humans + yolo
data = pd.concat([HOB_df,HY_df],ignore_index=True)
fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MR_JI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Yolov5'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_HvHY.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MR_SI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Yolov5'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_HvHY.png'))
#plt.show()
plt.close()

#plot humans only vs humans + CP        
data = pd.concat([HO_df,HCP_df],ignore_index=True)
fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MR_JI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Cellpose2.0'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_HvHCP.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MR_SI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Cellpose2.0'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
pplt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_HvHCP.png'))
#plt.show()         
plt.close()
                
#plot humans only bbox vs humans + yolo
data = pd.concat([HOB_df,HY_df],ignore_index=True)
fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MPW_JI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Yolov5'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.title('Pairwise Mean')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_MPW_HvHY.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MPW_SI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Yolov5'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_MPW_HvHY.png'))
#plt.show()
plt.close()

#plot humans only vs humans + CP        
data = pd.concat([HO_df,HCP_df],ignore_index=True)
fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MPW_JI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Cellpose2.0'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_MPW_HvHCP.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MPW_SI',hue='IOUth')
plt.xticks([0,1],labels=['Humans Only','Humans+Cellpose2.0'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_MPW_HvHCP.png'))
#plt.show()        
plt.close()

#plot humans only vs humans + humans       
data = pd.concat([HO_df,HC_df],ignore_index=True)
fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MPW_JI',hue='IOUth')
plt.xticks([0,1],labels=['Human Readers','Humans Readers + Re-segment'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_MPW_HvHC.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MPW_SI',hue='IOUth')
plt.xticks([0,1],labels=['Human Readers','Humans Readers + Re-segment'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'SorensenIndex_MPW_HvHC.png'))
#plt.show()  
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MR_JI',hue='IOUth')
plt.xticks([0,1],labels=['Human Readers','Humans Readers + Re-segment'])
plt.xlabel('')
plt.ylabel('Jaccard Index')
plt.legend()
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(pdir,'JaccardIndex_MR_HvHC.png'))
#plt.show()
plt.close()

fig = plt.figure(figsize=(4,3),dpi=600)
sn.violinplot(data=data,x='NumReaders',y='MR_SI',hue='IOUth')
plt.xticks([0,1],labels=['Human Readers','Humans Readers + Re-segment'])
plt.xlabel('')
plt.ylabel('Sorensen Index')
plt.legend()
plt.ylim([0,1])
plt.savefig(os.path.join(pdir,'SorensenIndex_MR_HvHC.png'))
plt.show() 


for th in [0.25,0.5,0.75]:    
    
    HOdf = HO_df[HO_df['IOUth']==th] 
    HCPdf = HCP_df[HCP_df['IOUth']==th]   
    HOBdf = HOB_df[HOB_df['IOUth']==th] 
    HYdf = HY_df[HY_df['IOUth']==th] 
    HCdf = HC_df[HC_df['IOUth']==th]

    print('*****MPW vs MR*****')
    JI_stat,JI_pval = mannwhitneyu(HOdf['MR_JI'],HOdf['MPW_JI'])
    SI_stat,SI_pval = mannwhitneyu(HOdf['MR_SI'],HOdf['MPW_SI'])
    
    print('HumansOnly JI MR vs. MPW, Mann Whitney U p-val: at IOU:',th,':',JI_pval*3)
    print('HumansOnly SI MR vs. MPW, Mann Whitney U p-val: at IOU:',th,':',SI_pval*3)
    print('')

    print('*****Multi-reader*****')
    HOBJISE = sem(HOBdf['MR_JI'])
    HOBJIH = HOBJISE*t.ppf(1.95/2,49)
    HOBSISE = sem(HOBdf['MR_SI'])
    HOBSIH = HOBSISE*t.ppf(1.95/2,49)
    HYJISE = sem(HYdf['MR_JI'])
    HYJIH = HYJISE*t.ppf(1.95/2,49)
    HYSISE = sem(HYdf['MR_SI'])
    HYSIH = HYSISE*t.ppf(1.95/2,49)
    #humans v humans+cp MR   
    print('Human Readers MR Jaccard at IOU:',th,':',HOBdf['MR_JI'].mean(),'[',HOBdf['MR_JI'].mean()-HOBJIH,',',HOBdf['MR_JI'].mean()+HOBJIH,']')
    print('Human Readers MR Sorensen at IOU:',th,':',HOBdf['MR_SI'].mean(),'[',HOBdf['MR_SI'].mean()-HOBSIH,',',HOBdf['MR_SI'].mean()+HOBSIH,']')
    print('Human Readers + Yolov5 MR Jaccard at IOU:',th,':',HYdf['MR_JI'].mean(),'[',HYdf['MR_JI'].mean()-HYJIH,',',HYdf['MR_JI'].mean()+HYJIH,']')
    print('Human Readers + Yolov5 MR Sorensen at IOU:',th,':',HYdf['MR_SI'].mean(),'[',HYdf['MR_SI'].mean()-HYSIH,',',HYdf['MR_SI'].mean()+HYSIH,']')
    
    print('')
    YoloLBJI,MIJI = NI_test(list(HYdf['MR_JI']),list(HOBdf['MR_JI']),0.1,equal_var=False,increase_good=True)
    YoloLBSI,MISI = NI_test(list(HYdf['MR_SI']),list(HOBdf['MR_SI']),0.1,equal_var=False,increase_good=True)
    
    N1 = len(HYdf['MR_JI'])
    N2 = len(HOBdf['MR_JI'])
    Jdiffs = []
    Sdiffs = []
    for x in HYdf['MR_JI']:
        for y in HOBdf['MR_JI']:
            Jdiffs.append(x-y)
    for x in HYdf['MR_SI']:
        for y in HOBdf['MR_SI']:
            Sdiffs.append(x-y)
            
    Jdiffs.sort()
    Sdiffs.sort()
    
    LB = int((N1*N2/2)-1.96*np.sqrt(N1*N2*(N1+N2+1)/12))
    UB = int((N1*N2/2)+1.96*np.sqrt(N1*N2*(N1+N2+1)/12))
    
    yJI_stat,yJI_pval = mannwhitneyu(HYdf['MR_JI'],HOBdf['MR_JI'],alternative='less')
    ySI_stat,ySI_pval = mannwhitneyu(HYdf['MR_SI'],HOBdf['MR_SI'],alternative='less')
    
    print('Humans vs. Humans + Yolov5 MR Jaccard, Mann Whitney U p-val: at IOU:',th,':',yJI_pval,'95%CI:',[Jdiffs[LB],Jdiffs[UB]])
    print('Lower 90% bound of difference in performance:',YoloLBJI,'Threshold:',-1*np.abs(MIJI))
    print('Humans vs. Humans + Yolov5 MR Sorensen, Mann Whitney U p-val: at IOU:',th,':',ySI_pval,'95%CI:',[Sdiffs[LB],Sdiffs[UB]])
    print('Lower 90% bound of difference in performance:',YoloLBSI,'Threshold:',-1*np.abs(MISI))
    print('')
    
    HOJISE = sem(HOdf['MR_JI'])
    HOJIH = HOJISE*t.ppf(1.95/2,49)
    HOSISE = sem(HOdf['MR_SI'])
    HOSIH = HOSISE*t.ppf(1.95/2,49)
    HCPJISE = sem(HCPdf['MR_JI'])
    HCPJIH = HCPJISE*t.ppf(1.95/2,49)
    HCPSISE = sem(HCPdf['MR_SI'])
    HCPSIH = HCPSISE*t.ppf(1.95/2,49)
    #humans v humans+cp MR   
    print('Human Readers MR Jaccard at IOU:',th,':',HOdf['MR_JI'].mean(),'[',HOdf['MR_JI'].mean()-HOJIH,',',HOdf['MR_JI'].mean()+HOJIH,']')
    print('Human Readers MR Sorensen at IOU:',th,':',HOdf['MR_SI'].mean(),'[',HOdf['MR_SI'].mean()-HOSIH,',',HOdf['MR_SI'].mean()+HOSIH,']')
    print('Human Readers + Cellpose2.0 MR Jaccard at IOU:',th,':',HCPdf['MR_JI'].mean(),'[',HCPdf['MR_JI'].mean()-HCPJIH,',',HCPdf['MR_JI'].mean()+HCPJIH,']')
    print('Human Readers + Cellpose2.0 MR Sorensen at IOU:',th,':',HCPdf['MR_SI'].mean(),'[',HCPdf['MR_SI'].mean()-HCPSIH,',',HCPdf['MR_SI'].mean()+HCPSIH,']')
    
    print('')
    
    CPLBJI,MIJI = NI_test(list(HCPdf['MR_JI']),list(HOdf['MR_JI']),0.1,equal_var=False,increase_good=True)
    CPLBSI,MISI = NI_test(list(HCPdf['MR_SI']),list(HOdf['MR_SI']),0.1,equal_var=False,increase_good=True)
    
    N1 = len(HCPdf['MR_JI'])
    N2 = len(HOdf['MR_JI'])
    Jdiffs = []
    Sdiffs = []
    for x in HCPdf['MR_JI']:
        for y in HOdf['MR_JI']:
            Jdiffs.append(x-y)
    for x in HCPdf['MR_SI']:
        for y in HOdf['MR_SI']:
            Sdiffs.append(x-y)
    Jdiffs.sort()
    Sdiffs.sort()
    
    LB = int((N1*N2/2)-1.96*np.sqrt(N1*N2*(N1+N2+1)/12))
    UB = int((N1*N2/2)+1.96*np.sqrt(N1*N2*(N1+N2+1)/12))
    
    cpJI_stat,cpJI_pval = mannwhitneyu(HCPdf['MR_JI'],HOdf['MR_JI'],alternative='less')
    cpSI_stat,cpSI_pval = mannwhitneyu(HCPdf['MR_SI'],HOdf['MR_SI'],alternative='less')
    
    print('Humans vs. Humans + Cellpose2.0 MR Jaccard, Mann Whitney U p-val: at IOU:',th,':',cpJI_pval*3,'95%CI:',[Jdiffs[LB],Jdiffs[UB]])
    print('Lower 90% bound of difference in performance:',CPLBJI,'Threshold:',-1*np.abs(MIJI))
    print('Humans vs. Humans + Cellpose2.0 MR Sorensen, Mann Whitney U p-val: at IOU:',th,':',cpSI_pval*3,'95%CI:',[Sdiffs[LB],Sdiffs[UB]])
    print('Lower 90% bound of difference in performance:',CPLBSI,'Threshold:',-1*np.abs(MISI))
    print('')
    
    print('*****Pairwise Mean*****')
    #humans bbox MPW    
    HOBJISE = sem(HOBdf['MPW_JI'])
    HOBJIH = HOBJISE*t.ppf(1.95/2,49)
    HOBSISE = sem(HOBdf['MPW_SI'])
    HOBSIH = HOBSISE*t.ppf(1.95/2,49)
    print('Human Readers Bbox MPW Jaccard at IOU:',th,':',HOBdf['MPW_JI'].mean(),'[',HOBdf['MPW_JI'].mean()-HOBJIH,',',HOBdf['MPW_JI'].mean()+HOBJIH,']')
    print('Human Readers Bbox MPW Sorensen at IOU:',th,':',HOBdf['MPW_SI'].mean(),'[',HOBdf['MPW_SI'].mean()-HOBSIH,',',HOBdf['MPW_SI'].mean()+HOBSIH,']')
    
    #humans MPW    
    HOJISE = sem(HOdf['MPW_JI'])
    HOJIH = HOJISE*t.ppf(1.95/2,49)
    HOSISE = sem(HOdf['MPW_SI'])
    HOSIH = HOSISE*t.ppf(1.95/2,49)
    print('Human Readers MPW Jaccard at IOU:',th,':',HOdf['MPW_JI'].mean(),'[',HOdf['MPW_JI'].mean()-HOJIH,',',HOdf['MPW_JI'].mean()+HOJIH,']')
    print('Human Readers MPW Sorensen at IOU:',th,':',HOdf['MPW_SI'].mean(),'[',HOdf['MPW_SI'].mean()-HOSIH,',',HOdf['MPW_SI'].mean()+HOSIH,']')
    
    print('')
    
    
   
