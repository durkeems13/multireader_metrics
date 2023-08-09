import os
import argparse
import numpy as np
from pycocotools import mask as pycocomask
from imagej_tiff_meta import TiffFile
from tifffile import imwrite

def process_single_overlay(overlay):
    new_ch = [1]
    new_pts = [overlay]
    pts_by_ch = []
    for i in range(max(new_ch)):
        pts_by_ch.append([x for x,y in zip(new_pts,new_ch) if y==i])
    return pts_by_ch

def process_overlays(overlays):
    overlays = [x for x in overlays if x['roi_type']!=1]
    p1 = lambda x,key : [x[i][key] for i in range(len(x))
            if 'multi_coordinates' in x[i].keys()]
    kys = ['position','left','top','multi_coordinates']
    overlays = [p1(overlays,key) for key in kys]

    p2 = lambda x,y : [a for (a,b) in zip(x,y) if y != None]
    overlays = [p2(x,overlays[3])for x in overlays]
    
    overlays[3] = [x[0] for x in overlays[3]]

    new_pts = []
    new_ch = []
    for (c,left,top,x) in zip(*overlays):
        if x.ndim==2:
            x[:,0] += left
            x[:,1] += top
            new_pts.append(x)
            new_ch.append(c)
    if len(new_ch)==0:
        return []

    if max(new_ch)>0:
        pts_by_ch = []
        for i in range(max(new_ch)):
            pts_by_ch.append([x for x,y in zip(new_pts,new_ch) if y==i])
    else:
        pts_by_ch = [[x for x,y in zip(new_pts,new_ch)]]
    return pts_by_ch

def outlines_to_mask(overlays_by_ch,imh,imw,imc):
    im_mask = np.zeros([imh,imw,imc],dtype=np.uint8)
    for i,points in enumerate(overlays_by_ch):
        ch_mask = np.zeros([imh,imw],dtype=np.uint8)
        cellct = 1
        for j,pts in enumerate(points):
            if pts.shape[0] > 2:
                re = pycocomask.frPyObjects([pts.flatten()],imh,imw)
                mask = pycocomask.decode(re)
                im_mask[:,:,i] = np.where(im_mask[:,:,i]>mask[...,0]*cellct,im_mask[:,:,i],mask[...,0]*cellct)
                cellct+=1
    return im_mask

def get_from_overlay(imName):
    t = TiffFile(imName)
    if hasattr(t.pages[0],'imagej_tags'):
        if 'parsed_overlays' in t.pages[0].imagej_tags.keys():
            overlays = t.pages[0].imagej_tags.parsed_overlays
            processed_overlays = process_overlays(overlays)
        elif 'overlays' in t.pages[0].imagej_tags.keys():
            overlays = t.pages[0].imagej_tags.overlays
            processed_overlays = process_single_overlay(overlays)
        else:
            processed_overlays = []
    else:
        processed_overlays = []
    if len(np.shape(t.asarray()))==3:
        imc,imh,imw = np.shape(t.asarray())
    elif len(np.shape(t.asarray()))==2:
        imh,imw = np.shape(t.asarray())
        imc=1
    elif len(np.shape(t.asarray()))==4:
        imc,imh,imw,_=np.shape(t.asarray())
    else:
        print(np.shape(t.asarray()))
        print('Unexpected image size. Please check input data')
        return
    t.close()
    return processed_overlays,imh,imw,imc

# working (see final overlays for examples)
def workloop(gt_im,write_dir):
    imname = gt_im.split('/')[-1]
    overlays,im_h,im_w,im_c = get_from_overlay(gt_im)
    masks_by_ch = outlines_to_mask(overlays,im_h,im_w,im_c)
    np.save(os.path.join(write_dir,imname.replace('.tif','.npy')),masks_by_ch)
    
def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--rootdir",type=str,default='/'.join(os.getcwd().split('/')[:-1]),help="")
    parser.add_argument("--readdir",type=str,default='data',help="")
    parser.add_argument("--savedir",type=str,default='GT_npy',help="")

    args, unparsed = parser.parse_known_args()
    readdir = os.path.join(args.rootdir,args.readdir)
    savedir = os.path.join(args.rootdir,args.readdir,args.savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    ims = []
    datasets = os.listdir(readdir)
    datasets = [x for x in datasets if x.startswith('Dataset')]
    datasets = [x for x in datasets if os.path.isdir(os.path.join(readdir,x))]
    for dataset in datasets:
        print(dataset)
        if not os.path.exists(os.path.join(savedir,dataset)):
            os.makedirs(os.path.join(savedir,dataset))
        readers = os.listdir(os.path.join(readdir,dataset))
        for reader in readers:
            if reader=='Yolov5':
                continue
            if not os.path.exists(os.path.join(savedir,dataset,reader)):
                os.makedirs(os.path.join(savedir,dataset,reader))
            imgs = os.listdir(os.path.join(readdir,dataset,reader))
            for img in imgs:
                if os.path.exists(os.path.join(savedir,dataset,reader,img.replace('.tif','.npy'))):
                    continue
                ims.append([os.path.join(readdir,dataset,reader,img),os.path.join(savedir,dataset,reader)])
    print(len(ims))
    for im in ims:
        workloop(im[0],im[1])

if __name__ == '__main__':
    main()
