#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 20:09:54 2023

@author: durkeems
"""
import warnings
warnings.simplefilter('ignore')
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

#SET PLOTTING PARAMETERS
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE, family='sans', variant='small-caps')         # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir',type=str,default='/'.join(os.getcwd().split('/')[:-1]),help='')
parser.add_argument('--agr_dir',type=str,default='data/agreement_matrices/',help='directory to save output to')
parser.add_argument('--save_dir',type=str,default='plots/heatmaps',help='directory to save output to')

args,unparsed=parser.parse_known_args()

adir = os.path.join(args.root_dir,args.agr_dir)
sdir = os.path.join(args.root_dir,args.save_dir)

if not os.path.exists(sdir):
    os.makedirs(sdir)

ags = os.listdir(adir)

Hags = [x for x in ags if 'CellPose' not in x]
Hags = [x for x in Hags if 'Yolo' not in x]
Hags = [x for x in Hags if 'Bbox' not in x]
Hags = [x for x in Hags if '2' not in x.split('_')[-2]]

hmap = np.zeros([5,5])

readers = ['ReaderE','ReaderA','ReaderB','ReaderC','ReaderD']
ai = ['Yolov5','CellPose']

for i in range(len(readers)):
    alist = [x for x in Hags if x.split('_')[-4]==readers[i]]
    for j in range(len(readers)):
        alist2 = [x for x in alist if x.split('_')[-2]==readers[j]]
        JI=[]
        for a in alist2:
            f = np.load(os.path.join(adir,a),allow_pickle=True).item()
            f = f['AgreementMatrix']
            f = np.where(f>=0.5,f,0)
            r1,r2 = np.shape(f)
            tp = np.count_nonzero(f)
            fp = r1-tp
            fn = r2-tp
            JI.append(tp/(tp+fp+fn))
        hmap[j,i]=np.mean(JI)

plt.figure(figsize=(3,3),dpi=600)
sn.heatmap(hmap,annot=True,vmin=0,vmax=1)
plt.xticks(ticks=[0.5,1.5,2.5,3.5,4.5],rotation=90,labels=['Reader1','Reader2','Reader3','Reader4','Reader5'])
plt.yticks(ticks=[0.5,1.5,2.5,3.5,4.5],rotation=-0,labels=['Reader1','Reader2','Reader3','Reader4','Reader5'])
plt.title('Mean Jaccard Index')
plt.tight_layout()
plt.savefig(os.path.join(sdir,'HumanvHumanHeatmap.png'))
plt.show()

HCags = [x for x in ags if '2' not in x.split('_')[-2]]
HYags = [x for x in HCags if 'Yolov5' in x]
HYags = [x for x in HYags if 'Bbox' in x]

HCPags = [x for x in HCags if 'CellPose' in x]
HCPags = [x for x in HCPags if 'Bbox' not in x]

Hags2 = HCPags+HYags
Hags2 = [x for x in Hags2 if 'Yolov5_vs_CellPose' not in x]


hmap2 = np.zeros([5,2])

for i in range(len(readers)):
    alist = [x for x in Hags2 if x.split('_')[-4]==readers[i]]
    for j in range(len(ai)):
        alist2 = [x for x in alist if x.split('_')[-2]==ai[j]]
        JI=[]
        for a in alist2:
            f = np.load(os.path.join(adir,a),allow_pickle=True).item()
            f = f['AgreementMatrix']
            f = np.where(f>=0.5,f,0)
            r1,r2 = np.shape(f)
            tp = np.count_nonzero(f)
            fp = r1-tp
            fn = r2-tp
            JI.append(tp/(tp+fp+fn))
        hmap2[i,j]=np.mean(JI)

plt.figure(figsize=(3,3),dpi=600)
sn.heatmap(hmap2,annot=True,vmin=0,vmax=1)
plt.yticks(ticks=[0.5,1.5,2.5,3.5,4.5],rotation=0,labels=['Reader1','Reader2','Reader3','Reader4','Reader5'])
plt.xticks(ticks=[0.5,1.5],rotation=0,labels=['Yolov5','Cellpose2.0'])
plt.title('Mean Jaccard Index')
plt.tight_layout()
plt.savefig(os.path.join(sdir,'HumanvAIHeatmap.png'))
plt.show()
