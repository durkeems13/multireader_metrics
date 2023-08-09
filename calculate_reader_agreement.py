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
import multiprocessing as mp
from tqdm import tqdm
from itertools import combinations


def create_match_matrix(batch):
    im_comps = batch['ImageComps']
    aggdir = batch['AggDir']
    matchdir = batch['MatchDir']
    cname = batch['CompName']
    imname = '_'.join(im_comps[0].split('_')[:-4])
    
    if not os.path.exists(os.path.join(matchdir,cname)):
        try:
            os.makedirs(os.path.join(matchdir,cname))
        except:
            print(cname,'folder created')
    
    R1List = [x.split('_')[-4] for x in im_comps]
    R2List = [x.split('_')[-2] for x in im_comps]
    R1List,cts = np.unique(R1List,return_counts=True)
    cts2=cts.argsort()[::-1]
    R1List = R1List[cts2]
    R1List = [str(x) for x in R1List]
    
    r2_u = list(set(R2List)-set(R1List))
    r2_u = [str(x) for x in r2_u]
    all_readers = R1List+r2_u
    
    ths = [0.25,0.5,0.75]
    for th in ths:
        csvname = os.path.join(matchdir,cname,imname+'_TH_'+str(th)+'.csv')
        #if os.path.exists(csvname):
        #    continue
        match_df = pd.DataFrame(columns=all_readers)
        for i,R1 in enumerate(R1List):
            R1comps = [x for x in im_comps if x.split('_')[-4]==R1]
            r1_matches = list(match_df[R1])
            r1_matches = [x for x in r1_matches if x!=None]
            comp_mats = {}
            for j,comp in enumerate(R1comps):
                compdict = np.load(os.path.join(aggdir,comp),allow_pickle=True).item()
                R2 = comp.split('_')[-2]
                agg = compdict['AgreementMatrix']
                agg2 = np.where(agg>th,agg,0)
                comp_mats.update({R1+'_'+R2:agg2})
            R1cells,_=np.shape(agg2)
            R1cells=np.arange(0,R1cells)
            newR1cells = list(set(R1cells)-set(r1_matches))
            R1cells.sort()
            r1_matches.sort()
            newR1cells.sort()
            for cell in newR1cells:
                cdict = {}
                for R in all_readers:
                    cdict[R]=[None]
                cdict[R1]=[cell]
                for ky in comp_mats.keys():
                    R2=ky.split('_')[-1]
                    aggmat = comp_mats[ky]
                    r2_inds = list(aggmat[cell,:])
                    r2_match = [i for i,x in enumerate(r2_inds) if x > 0]
                    if len(r2_match)==0:
                        cdict[R2]=[None]
                    else:
                        cdict[R2]=r2_match

                cdf = pd.DataFrame.from_dict(cdict)
                match_df = pd.concat([match_df,cdf],ignore_index=True)
        match_df.to_csv(csvname)

    
def create_pairwise_agreement_matrix(batch):
    
    r1 = batch['Reader1']
    r2 = batch['Reader2']
    rdir = batch['ImageDir']
    imname = batch['ImageName']
    aggdir = batch['SaveDir']
    aggname = imname.split('.')[0]+'_'+r1.split('/')[-1]+'_vs_'+r2.split('/')[-1]+'_agreement.npy'
    aggpath = os.path.join(aggdir,aggname)

    
    bboxname = imname.split('.')[0]+'_'+r1.split('/')[-1]+'_vs_'+r2.split('/')[-1]+'_Bbox-agreement.npy'
    bboxpath = os.path.join(aggdir,bboxname)
    
    areath=100
    
    r1im=np.load(os.path.join(rdir,r1,imname))
    if len(np.shape(r1im))==3:
        r1im = r1im[:,:,0]
    r,c = np.shape(r1im)
    r1cells = np.unique(r1im)
    r1cells = np.delete(r1cells,0)
    
    if r2=='Yolov5':
        label_path = os.path.join(rdir.replace('/GT_npy',''),r2,'labels',imname.replace('.npy','.txt'))
        with open(label_path,'r') as f:
            cells=f.readlines()
        f.close()
        r2cells=[]
        for cell in cells:
            _,cc,cr,cw,ch,_ = cell.split(' ')
            fixed_cell = [int(float(cc)*c),int(float(cr)*r),int(float(cw)*c),int(float(ch)*r)]
            fixed_cell2 = [[fixed_cell[1],fixed_cell[1]+fixed_cell[3]],[fixed_cell[0],fixed_cell[0]+fixed_cell[2]]]
            r2cells.append(fixed_cell2)
    else:
        r2im = np.load(os.path.join(rdir,r2,imname))
        r2cells=np.unique(r2im)
        r2cells=np.delete(r2cells,0)
    
    a_mat = np.zeros([len(r1cells),len(r2cells)])
    bbox_mat=np.zeros([len(r1cells),len(r2cells)])
    for i,r1cell in tqdm(enumerate(r1cells)):
        r1_mask = np.where(r1im==r1cell,1,0)
        cell1_ind = np.where(r1_mask==1)
        if len(cell1_ind[0])<areath:
            continue
        rmin1 = np.min(cell1_ind[0])
        rmax1 = np.max(cell1_ind[0])
        cmin1 = np.min(cell1_ind[1])
        cmax1 = np.max(cell1_ind[1])
        r1_bbox = np.zeros([r,c])
        r1_bbox[rmin1:rmax1,cmin1:cmax1]=1
        for j,r2cell in enumerate(r2cells):
            if r2=='Yolov5':
                r2_mask=np.zeros([r,c])
                r2_mask[r2cell[0][0]:r2cell[0][1],r2cell[1][0]:r2cell[1][1]]=1
                r1_mask[rmin1:rmax1,cmin1:cmax1]=1
            else:
                r2_mask = np.where(r2im==r2cell,1,0)
                if len(np.shape(r2_mask))==3:
                    r2_mask = r2_mask[:,:,0]
            cell2_ind = np.where(r2_mask==1)
            if len(cell2_ind[0])<areath:
                continue
            rmin2 = np.min(cell2_ind[0])
            rmax2 = np.max(cell2_ind[0])
            cmin2 = np.min(cell2_ind[1])
            cmax2 = np.max(cell2_ind[1])
            r2_bbox = np.zeros([r,c])
            r2_bbox[rmin2:rmax2,cmin2:cmax2]=1
            if (rmin2 >= rmax1) or (rmax2 <= rmin1) or (cmax2 <= cmin1) or (cmin2 >= cmax1):
                continue
            
            ov = r1_mask+r2_mask
            u = np.sum(np.where(ov>0,1,0))
            io = np.sum(np.where(ov==2,1,0))
            iou = io/u
            a_mat[i,j]=iou
            
            bbox_ov = r1_bbox+r2_bbox
            bbox_u = np.sum(np.where(bbox_ov>0,1,0))
            bbox_io = np.sum(np.where(bbox_ov==2,1,0))
            bbox_iou = bbox_io/bbox_u
            bbox_mat[i,j]=bbox_iou
            
    if len(r1cells)>0 and len(r2cells)>0:
        aR = a_mat*(a_mat>=np.sort(a_mat,axis=0)[[-1],:])
        aC = a_mat*(a_mat>=np.sort(a_mat,axis=1)[:,[-1]])
        aggmat = np.sqrt(aR*aC)
    else:
        aggmat=a_mat.copy()

    agg = {'R1count':len(r1cells),'R2count':len(r2cells),'AgreementMatrix':aggmat}
    np.save(aggpath,agg,allow_pickle=True)
    
    if len(r1cells)>0 and len(r2cells)>0:
        baR = bbox_mat*(bbox_mat>=np.sort(bbox_mat,axis=0)[[-1],:])
        baC = bbox_mat*(bbox_mat>=np.sort(bbox_mat,axis=1)[:,[-1]])
        bboxaggmat = np.sqrt(baR*baC)
    else:
        bboxaggmat=bbox_mat.copy()
    bboxagg = {'R1count':len(r1cells),'R2count':len(r2cells),'AgreementMatrix':bboxaggmat}
    np.save(bboxpath,bboxagg,allow_pickle=True)
    

def compare_readers(imdir,aggdir,matchdir,comp):
    
    R1 = comp[0]
    R2 = comp[1]
    
    if 'Yolo' not in R1:
        ims = os.listdir(os.path.join(imdir,R1))
    else:
        ims = os.listdir(os.path.join(imdir,R2))
    
    num_workers = len(ims)
    batches = [{'Reader1':R1,'Reader2':R2,'ImageDir':imdir,'ImageName':x,'SaveDir':aggdir} for x in ims]
    pool = mp.Pool(num_workers)
    pool.map(create_pairwise_agreement_matrix,batches)
    pool.close()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',type=str,default='/'.join(os.getcwd().split('/')[:-1]),help='')
    parser.add_argument('--read_dir',type=str,default='data/GT_npy',help='validation label directory')
    parser.add_argument('--amat_dir',type=str,default='data/agreement_matrices',help='directory to save output to')
    parser.add_argument('--mmat_dir',type=str,default='data/match_matrices',help='directory to save output to')
    
    args,unparsed=parser.parse_known_args()

    imdir = os.path.join(args.root_dir,args.read_dir)
    amatdir = os.path.join(args.root_dir,args.amat_dir)
    mmatdir = os.path.join(args.root_dir,args.mmat_dir)
    
    if not os.path.exists(amatdir):
        os.makedirs(amatdir)
    if not os.path.exists(mmatdir):
        os.makedirs(mmatdir)
    
    datasets = os.listdir(imdir)
    reader_list = []
    for dataset in datasets:
        readers = os.listdir(os.path.join(imdir,dataset))
        readers = [dataset+'/'+x for x in readers if not x.endswith('1')]
        reader_list.extend(readers)
       
    comps = list(combinations(reader_list,2))
    for comp in comps:
        compare_readers(imdir,amatdir,mmatdir,comp)
       
    aggmats = os.listdir(amatdir)
    
    for case in [0,1,2,3]:
        if case==0:
            ### Humans vs DCNNs
            readerList1=['ReaderA','ReaderB','ReaderC','ReaderD','ReaderE'] #humans only
            readerList2=readerList1+['Yolov5']
            readerList3=readerList1+['CellPose']
            readerList4=readerList1+['ReaderB2','ReaderD2','ReaderE2']
            
            readerLists = [readerList1,readerList1,readerList2,readerList3,readerList4]
            readerListNames = ['HumansOnly','HumansOnly-Bbox','Humans+Yolov5','Humans+Cellpose','HumanConsistency']
        elif case==1:
            ### Reader Ablation
            rlist1 = ['ReaderA','ReaderB','ReaderC','ReaderD']
            rlist2 = ['ReaderA','ReaderB','ReaderC','ReaderE']
            rlist3 = ['ReaderA','ReaderB','ReaderD','ReaderE']
            rlist4 = ['ReaderA','ReaderC','ReaderD','ReaderE']
            rlist5 = ['ReaderB','ReaderC','ReaderD','ReaderE']
            
            readerLists = [rlist1,rlist2,rlist3,rlist4,rlist5]
            readerListNames = ['HumansOnly-NoE','HumansOnly-NoD','HumansOnly-NoC','HumansOnly-NoB','HumansOnly-NoA']
        elif case==2:
            ### Reader ablation with CP
            rlist1 = ['ReaderA','ReaderB','ReaderC','ReaderD','CellPose']
            rlist2 = ['ReaderA','ReaderB','ReaderC','ReaderE','CellPose']
            rlist3 = ['ReaderA','ReaderB','ReaderD','ReaderE','CellPose']
            rlist4 = ['ReaderA','ReaderC','ReaderD','ReaderE','CellPose']
            rlist5 = ['ReaderB','ReaderC','ReaderD','ReaderE','CellPose']
            
            readerLists = [rlist1,rlist2,rlist3,rlist4,rlist5]
            readerListNames = ['Humans+CellPose-NoE','Humans+CellPose-NoD','Humans+CellPose-NoC','Humans+CellPose-NoB','Humans+CellPose-NoA']
        elif case==3:
            ### Reader ablation with Yolo
            rlist1 = ['ReaderA','ReaderB','ReaderC','ReaderD','Yolov5']
            rlist2 = ['ReaderA','ReaderB','ReaderC','ReaderE','Yolov5']
            rlist3 = ['ReaderA','ReaderB','ReaderD','ReaderE','Yolov5']
            rlist4 = ['ReaderA','ReaderC','ReaderD','ReaderE','Yolov5']
            rlist5 = ['ReaderB','ReaderC','ReaderD','ReaderE','Yolov5']
            
            readerLists = [rlist1,rlist2,rlist3,rlist4,rlist5]
            readerListNames = ['Humans+Yolov5-NoE','Humans+Yolov5-NoD','Humans+Yolov5-NoC',
                           'Humans+Yolov5-NoB','Humans+Yolov5-NoA']
        for idx,rList in enumerate(readerLists):
            compname = readerListNames[idx]
            rcomp_aggmats = [x for x in aggmats if (x.split('_')[-2] in rList) and (x.split('_')[-4] in rList)]
            if ('Yolo' in compname) or ('Bbox' in compname):
                rcomp_aggmats=[x for x in rcomp_aggmats if 'Bbox' in x]
            ims = ['_'.join(x.split('_')[:-4]) for x in rcomp_aggmats]
            ims = np.unique(ims)
            imbatches = []
            for im in ims:
                imbatch = [x for x in rcomp_aggmats if im in x]
                imbatches.append(imbatch)
            batches = [{'AggDir':amatdir,'MatchDir':mmatdir,'CompName':compname,'ImDir':imdir,'ImageComps':x} for x in imbatches]
            num_workers = len(batches)
            pool = mp.Pool(num_workers)
            pool.map(create_match_matrix,batches)
            pool.close()
        

    
if __name__=='__main__':
    main()
