# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.models.utils import ExtractProcess, knn_classifier
from mmselfsup.utils import get_root_logger
from numpy.linalg import norm
import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
import scipy.stats as stats


def parse_args():
    parser = argparse.ArgumentParser(description='homogeneity calculation evaluation')
    parser.add_argument('--config', default='configs/selfsup/simsiam/simsiam_resnet18_8xb32-coslr-400e_sem.py',help='train config file path')
    parser.add_argument('--checkpoint', default='work_dirs/selfsup/simsiam_resnet18_8xb32-coslr-400e_sem/epoch_400.pth', help='checkpoint file')
    parser.add_argument(
        '--dataset-config',
        default='configs/benchmarks/classification/knn_imagenet.py',
        help='knn dataset config file path')
    parser.add_argument(
        '--work-dir', type=str, default=None, help='the dir to save results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # KNN settings
    
    parser.add_argument(
        '--use-cuda',
        default=True,
        type=bool,
        help='Store the features on GPU. Set to False if you encounter OOM')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def create_crops(image_size,gap,whole_image_path,tag='o',x_num=None,y_num=None):
    save_folder='data_collect/crops/crops'
    whole_image=cv2.imread(whole_image_path)
#     if 'DPsteel_' in whole_image_path:
#         whole_image=whole_image[0:2048,:,:]
#     elif 'micrograph' in whole_image_path:
#         whole_image=whole_image[0:720,:,:]
    w,h,_=whole_image.shape
    if x_num is None:
        x_num=int(np.floor((w-image_size)/gap)+1-4) #3 or 5 related
        y_num=int(np.floor((h-image_size)/gap)+1-4) #3 or 5 related
    
    for xx in range(x_num):
        for yy in range(y_num):
            if tag=='o':
                x_begin=xx*gap+2*gap #3 or 5 related
                y_begin=yy*gap+2*gap #3 or 5 related
            else:
                x_begin=xx*gap
                y_begin=yy*gap
            x_end=x_begin+image_size
            y_end=y_begin+image_size
            crop=copy.deepcopy(whole_image[x_begin:x_end,y_begin:y_end,:])
            if tag=='mag':
                path=os.path.join(save_folder,whole_image_path.split('/')[-1].split('.')[0]+'_'+str(xx)+'_'+str(yy)+'_mag'+'.png')
            else:
                path=os.path.join(save_folder,whole_image_path.split('/')[-1].split('.')[0]+'_'+str(xx)+'_'+str(yy)+'.png')
            cv2.imwrite(path,crop)
    if tag=='mag':
        save_img=copy.deepcopy(whole_image)
        save_img=save_img[:(x_num-1)*gap+image_size,:(y_num-1)*gap+image_size,:]
        save_img=cv2.rectangle(save_img, 
                               (int((image_size-gap)/2),int((image_size-gap)/2)),
                               (int((y_num-1)*gap+image_size-(image_size-gap)/2),int((x_num-1)*gap+image_size-(image_size-gap)/2)),
                              (0,0,0),
                              5)
        cv2.imwrite(whole_img_name+'_rec.tif',save_img)
        cv2.imwrite(whole_img_name+'_new.tif',whole_image[:(x_num-1)*gap+image_size,:(y_num-1)*gap+image_size,:])
    return x_num,y_num
    

def main(whole_img_name):
    save_folder='data_collect/crops/crops'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        files=os.listdir(save_folder)
        for file in files:
            os.remove(os.path.join(save_folder,file))
            
    gap=50
    
    x_num,y_num=create_crops(image_size=int(gap*5),gap=gap,whole_image_path='data_collect/'+whole_img_name,tag='mag') #3 or 5 related
    _,_=create_crops(image_size=gap,gap=gap,whole_image_path='data_collect/'+whole_img_name,tag='o',x_num=x_num,y_num=y_num)
    whole_img_name=whole_img_name.split('.')[0]
    
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir and init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    knn_work_dir = osp.join(cfg.work_dir, 'knn/')
    mmcv.mkdir_or_exist(osp.abspath(knn_work_dir))
    log_file = osp.join(knn_work_dir, f'knn_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset_train = build_dataset(dataset_cfg.data.train)
#     print(dataset_train.data_source.data_infos)
    data_infos = dataset_train.data_source.data_infos
    img_filenames=[]
    for i in range(len(data_infos)):
        img_filenames.append(data_infos[i]['img_info']['filename'].split('/')[1])
#     print(img_filenames)
#     print(dataset_train.data_infos)
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loader_train = build_dataloader(
        dataset_train,
        samples_per_gpu=dataset_cfg.data.samples_per_gpu,
        workers_per_gpu=dataset_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    

    # build the model
    model = build_algorithm(cfg.model)
    model.init_weights()

    # model is determined in this priority: init_cfg > checkpoint > random
    if hasattr(cfg.model.backbone, 'init_cfg'):
        if getattr(cfg.model.backbone.init_cfg, 'type', None) == 'Pretrained':
            logger.info(
                f'Use pretrained model: '
                f'{cfg.model.backbone.init_cfg.checkpoint} to extract features'
            )
    elif args.checkpoint is not None:
        logger.info(f'Use checkpoint: {args.checkpoint} to extract features')
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        logger.info('No pretrained or checkpoint is given, use random init.')

    
    model = MMDataParallel(model, device_ids=[0])
    
    model.eval()
    # build extraction processor and run
    extractor = ExtractProcess()
    train_feats = extractor.extract(
        model, data_loader_train, distributed=distributed)['feat']


    print(train_feats.shape)
#     print(img_filenames)
    or_keep_index=[]
    mag_keep_index=[]
    for i in range(len(img_filenames)):
        if 'mag' not in img_filenames[i]:
            or_keep_index.append(i)
        else:
            mag_keep_index.append(i)   
    or_train_feats=train_feats[or_keep_index,:]
    mag_train_feats=train_feats[mag_keep_index,:]
    
    
    zoom_cosine_total=np.zeros(or_train_feats.shape[0])
    for i in range(or_train_feats.shape[0]):
        feat_i=or_train_feats[i,:]
        feat_j=mag_train_feats[i,:]
        cosine = np.dot(feat_i,feat_j)/(norm(feat_i)*norm(feat_j))
#                 print("Cosine Similarity:", cosine)
        zoom_cosine_total[i]=cosine
    print(whole_img_name)
    print('zoom mean=',np.mean(zoom_cosine_total))

    cosine_dic=[]
    or_img_filenames=np.array(img_filenames)[or_keep_index]
    max_zoom_cosine=np.amax(zoom_cosine_total)
    min_zoom_cosine=np.amin(zoom_cosine_total)
    for i in range(len(or_img_filenames)):
        img_filename=or_img_filenames[i]
        temp=img_filename.split('.')[0].split('_')
#         print(temp)
        xx=int(temp[-2])
        yy=int(temp[-1])
#         if zoom_cosine_total[i]>(0.9*(max_zoom_cosine-min_zoom_cosine)+min_zoom_cosine):
#             print(temp,zoom_cosine_total[i])
#         if zoom_cosine_total[i]<(0.1*(max_zoom_cosine-min_zoom_cosine)+min_zoom_cosine):
#             print(temp,zoom_cosine_total[i])
#         print(xx,yy)
        cosine_dic.append({'x':xx,'y':yy,'cos':zoom_cosine_total[i]})
    maxx=int(np.max([item['x'] for item in cosine_dic])+1)
    maxy=int(np.max([item['y'] for item in cosine_dic])+1)
    cosine_matrix=np.zeros((maxx,maxy))
    for item in cosine_dic:
            cosine_matrix[item['x'],item['y']]=item['cos']
#     print(cosine_matrix)
    fig2,axis2=plt.subplots(figsize=(10,5))
    a=axis2.imshow(cosine_matrix,cmap="hot",vmin=0.55,vmax=0.95)
    cb=plt.colorbar(a,ax=axis2)
   
    
    plt.xticks([])
    plt.yticks([])
    plt.savefig('zoom_gap'+str(gap)+'_'+whole_img_name+'.png')
    
    plt.close()

if __name__ == '__main__':
    whole_img_names=[item for item in os.listdir('data_collect/') if item.endswith('.jpeg') or item.endswith('.tif') or item.endswith('png')]
#     whole_img_names=['micrograph471-crop-spheroidite.png','micrograph679-crop-spheroidite.png']
    for whole_img_name in whole_img_names:
        main(whole_img_name)
