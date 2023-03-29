import pandas as pd
import numpy as np
import os
import glob
import torch
import cv2
import random
import tqdm
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.transforms  import RandomChoice
from torchvision.transforms  import functional as F
from tensorboardX import SummaryWriter
from PIL import Image
import argparse
from data_process import Resize,RandomCropChannel,RandomHorizontalFlip,RandomVerticalFlip,Normalization,ToTensor,RandomChoice
from data_process import MRDataset,Compose, RandomAffine,Swap_img_array
from model import Transformer_2d
from metrics import AUC,ACC,fc_false_positive,fc_false_negative,yuedenindex,AsymmetricLossOptimized,yuedenindex_m,
import torch.multiprocessing as mp

ap = argparse.ArgumentParser()  # cell_detect
ap.add_argument("--csvpath", default="./RAIN/csv", type=str)
ap.add_argument("--weightpath", default="./RAIN/weights", type=str)
ap.add_argument("--loadweight", default="",type=str)
ap.add_argument("--date", default="y-m-d",type=str)
ap.add_argument("--version", default="vs12",type=str)
ap.add_argument("--gpu", default="0,1", type=str)
ap.add_argument("--lr", default=1e-5, type=float)
ap.add_argument("--min_lr", default=1e-5, type=float)
ap.add_argument("--weight_decay", default=0, type=float)
ap.add_argument("--epochs", default=120, type=int)
ap.add_argument("--batch_size", default=6, type=int)
ap.add_argument("--samplefactor", default=1.0, type=float)
ap.add_argument("--num_classes",default=2,type=int)
ap.add_argument("--subsample",default=0,type=int)
ap.add_argument("--port", default="23456", type=str)
ap.add_argument("--num_workers",default=12,type=int)
ap.add_argument("--dcmnum", default=16,type=int)
ap.add_argument("--samplestyle",default="2*min", type=str)
ap.add_argument("--gamma_neg", default=1.44, type=float)
ap.add_argument("--gamma_pos", default=1, type=float)
ap.add_argument("--clip", default=0.1, type=float)
ap.add_argument("--backbone_name",default="resnet18", type=str)
ap.add_argument("--image_size",default=448, type=int)
ap.add_argument("--d_model", default=256, type=int)
ap.add_argument("--tf_n_layers", default=4, type=int)
ap.add_argument("--tf_n_head", default=6, type=int)

torch.cuda.manual_seed_all(1) 
torch.manual_seed(1)
arg = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu


# transforms
train_transform=Compose(transforms=[ Resize(size=(arg.image_size,arg.image_size)),RandomCropChannel(length=arg.dcmnum,dataset="test"),
                               Swap_img_array(),RandomAffine(p=0.5),Swap_img_array(),RandomHorizontalFlip(),RandomVerticalFlip(), Normalization(),ToTensor()])
test_transform=Compose(transforms=[Resize((arg.image_size,arg.image_size)), RandomCropChannel(length=arg.dcmnum,dataset="test"),Normalization(),ToTensor()])


if len(arg.gpu.split(","))>1:
    torch.distributed.init_process_group(backend="nccl",init_method='tcp://localhost:'+arg.port,rank=0, world_size=1)

def SingleDataSet(csvpath, samplefactor,transforms=train_transform):
    dataset = MRDataset(csvpath=csvpath, dataset= "train", transforms=transforms, samplefactor=samplefactor,subsample=arg.subsample,samplestyle=arg.samplestyle)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if len(arg.gpu.split(",")) > 1 else None
    return dataset,sampler

def SingleDataLoader(csvpath, dataset , batch_size,transforms , samplefactor,subsample=0,shuffle=False, drop_last=False,):
    dataset = MRDataset(csvpath=csvpath, dataset=dataset, transforms=transforms, samplefactor=samplefactor,subsample=subsample)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if len(arg.gpu.split(",")) > 1 else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last= drop_last,pin_memory=True, num_workers=arg.num_workers,sampler=sampler)  # ,sampler=train_sampler
    return dataloader

def my_DataLoader(csvpath,batch_size, samplefactor =1):
    val_dataloader = SingleDataLoader(csvpath, "val" , batch_size, test_transform ,1,0,shuffle=False, drop_last=False)
    train_dataloader_test = SingleDataLoader(csvpath, "train", batch_size, test_transform, 1, 0, shuffle=False,drop_last=False)
    return  val_dataloader,train_dataloader_test


def calc_loss(output_cls,Ts):
    loss = AsymmetricLossOptimized(gamma_neg=arg.gamma_neg, gamma_pos=arg.gamma_pos, clip=arg.clip)(output_cls,Ts.float())
    return loss
    # return train_dataloader,None,None

def train(csvpath,eventpath,weightpath,batch_size, samplefactor = 1,,num_init_features=64,
                num_classes=2,gpu="0",lr=1e-4,min_lr=5e-5,weight_decay=0.0,ad="default"):

    val_dataloader,  train_dataloader_test = my_DataLoader(csvpath, batch_size,samplefactor)
    if arg.samplestyle=="max":
        traindataset, trainsampler = SingleDataSet(csvpath, samplefactor, train_transform)
        train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=(trainsampler is None),drop_last=True, pin_memory=True, num_workers=arg.num_workers, sampler=trainsampler)

    Model = Transformer_2d(backbone_name=arg.backbone_name, image_size=arg.image_size, num_classes=num_classes, total_channels=arg.dcmnum, unit_channel=arg.unit_channel,
                     d_model=arg.d_model, d_inner=4*arg.d_model, n_layers=arg.tf_n_layers, n_head=arg.tf_n_head, d_k=64, d_v=64 dropout=0.1 pretrained=1,
                               backbone_feature_dim=None,pool="cls",feature_patch_d=2,feature_patch_h=1)
    if arg.loadweight:
        print("loadweght::...",arg.loadweight)
        Model.load_state_dict(torch.load(arg.loadweight))
    Model = Model.cuda()
    if len(gpu.split(","))>1:
        Model = nn.parallel.DistributedDataParallel(Model)
    if arg.modelname!="tf_2d":
        params = Model.parameters()
    else:
        backbone_params = list(map(id, Model.module.backbone.parameters()  if len(gpu.split(","))>1 else Model.backbone.parameters() ))
        encoder_params = filter(lambda p: id(p) not in backbone_params ,Model.parameters())
        params = [  {"params": Model.module.backbone.parameters()  if len(gpu.split(","))>1 else Model.backbone.parameters(), "lr": arg.lr_bb},{"params": encoder_params},]
    optimizer = optim.AdamW(params=params, lr=lr , weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=1, min_lr=min_lr ,eps=1e-8)

    flags ="T1C"
    tmpname=os.path.join("_".join(,"batchsize_{}_initlr_{}".format(str(batch_size ), str(lr)), arg.loss_style, flags)
    batch_num=0
    init_epoch=0
    init_epoch=0
    if arg.loadweight :
        loadweight_epoch=int(os.path.basename(arg.loadweight).split("_")[0])
        init_epoch= loadweight_epoch+1

    for epoch in range(init_epoch,arg.epochs):
        if arg.samplestyle != "max":
            traindataset, trainsampler = SingleDataSet(csvpath, samplefactor, train_transform)
            train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=(trainsampler is None),drop_last=True, pin_memory=True, num_workers=arg.num_workers,sampler=trainsampler)
        if trainsampler is not None:
            trainsampler.set_epoch(epoch)  ## 更新数据分布
        Model.train()
        all_loss_cls = []
        all_predict_cls_1 = []
        all_true_cls_1 = []
        for batch_id,(data,label,seriesname) in tqdm.tqdm(enumerate(train_dataloader)):
            data,label,seriesname=data.float().cuda(),label.float().cuda(),seriesname.float().cuda()

            data=data.squeeze(dim=1)
            output_cls=Model(data)
            loss=calc_loss(output_cls, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num+=1
            if batch_num % 50 == 0 or batch_num==2 or batch_num%1000==0:
                print(epoch, "batch,", batch_id, "arg.loss_style:", arg.loss_style,round(loss.item(),3))

            all_predict_cls_1.extend(output_cls.data.cpu().numpy()[:, 0].tolist())
            all_true_cls_1.extend(label.data.cpu().numpy()[:, 0].tolist())
            all_loss_cls.append(loss.item())

        predict_label_list = [[all_true_cls_1, all_predict_cls_1]]
        learn_rate_ec = optimizer.param_groups[-1]["lr"]
        weightpath_tmp = os.path.join(weightpath, "_".join("batchsize_{}_initlr_{}".format(str(batch_size), str(lr)), arg.loss_style, )
        if not os.path.exists(weightpath_tmp):
            os.makedirs(weightpath_tmp)# save model
        Data_r = []
        val_result, th_tra,= val(Model=Model, dataloader=val_dataloader, epoch=epoch,filename="val", outpath=weightpath_tmp, th=0.5)#
        train_result,_,=val_train(all_loss_cls,predict_label_list,epoch=epoch,filename="train",outpath=weightpath_tmp,th=th_tra)

        scheduler.step(val_result[-1])##
        Data_r.extend(train_result)
        Data_r.extend(val_result)

        columns_r_tmp=["auc", "acc_yd", "tpr_yd", "tnr_yd", "tpr*tnr_yd", "tpr-tnr_yd", "acc", "tpr", "tnr", "tpr*tnr","tpr-tnr" ]
        columns_r = ["_".join([s, k]) for s in ["train", "val"] for k in columns_r_tmp+["loss"]]
        weightpath_tmp = os.path.join(weightpath_tmp,str(epoch) + "_train_" +"_".join([str(round(r,3)) for r in train_result[:-1]])+"_val_" +"_".join([str(round(r,3)) for r in val_result[:-1]])+ ".pth")
        print("batch_num:", batch_num,"\n",weightpath_tmp)
        ## Experimental results
        Data_r = pd.DataFrame(np.array([epoch,th_tra]  + Data_r).reshape(1,-1), columns=["epoch","yuedengM_index"]   + columns_r)
        Data_r_path=os.path.join(os.path.dirname(weightpath_tmp), "_".join(["2.each_epoch_result_during_model_training",arg.date,flags,".xlsx"]))
        if os.path.exists(Data_r_path):
            Data_r_0=pd.read_excel(Data_r_path)
            Data_r=pd.concat([Data_r_0,Data_r],axis=0)
        Data_r.to_excel(Data_r_path,index=False)

        if len(gpu.split(",")) > 1:
            torch.save(Model.module.state_dict(), weightpath_tmp)
        else:
            torch.save(Model.state_dict(), weightpath_tmp)
        Model.train()


##
def val_train(all_loss_cls,predict_label_list,epoch,filename,outpath,th=None):

    val_loss = np.array(all_loss_cls).sum() / len(all_loss_cls)
    summarywriter.add_scalar("loss", val_loss, epoch)
    thresholds_dic={0:th,1:th2,2:th3}
    result = []
    for k,(all_true_cls,all_predict_cls) in enumerate(predict_label_list):
        indexs=np.arange(len(all_loss_cls)).reshape(-1,1)
        val_acc = ACC(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
        val_auc, fprs, tprs, thresholds= AUC(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
        val_fn = fc_false_negative(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
        val_fp = fc_false_positive(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
        tp, tn, th_tmp = yuedenindex_m(fprs, tprs, thresholds)
        if filename == "train":
            thresholds_dic[k]= th_tmp if 0.01 <= th_tmp <= 0.99 else  thresholds_dic[k]

        val_acc_th = ACC(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]), th=thresholds_dic[k])
        val_fn_th = fc_false_negative(y_true=list(np.array(all_true_cls)[indexs]),y_pred=list(np.array(all_predict_cls)[indexs]), th=thresholds_dic[k])
        val_fp_th = fc_false_positive(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]), th=thresholds_dic[k])
        result_tmp=[round(val_auc, 4), round(val_acc_th, 4), round(1 - val_fn_th, 4), round(1 - val_fp_th, 4),
            round((1 - val_fp_th) * (1 - val_fn_th), 4), round(abs((1 - val_fp_th) - (1 - val_fn_th)), 4),
            round(val_acc, 4), round(1 - val_fn, 4), round(1 - val_fp, 4), round((1 - val_fp) * (1 - val_fn), 4),round(abs((1 - val_fp) - (1 - val_fn)), 4)]
        result.extend(result_tmp)
    return result+[ val_loss], thresholds_dic[0]

def val(Model, dataloader,epoch,filename,outpath,th=None):

    batch_num = 0
    with torch.no_grad():
        Model.eval()
        batch_num = 0
        all_loss_cls = []
        all_predict_cls_1= []
        all_seriesname=[]
        all_true_cls_1 = []
        result=[]
        for batch_id, (data, label, seriesname) in tqdm.tqdm(enumerate(dataloader)):
            data, label, seriesname = data.float().cuda(), label.float().cuda(), seriesname.float().cuda()
            data=data.squeeze(dim=1)
            output_cls=Model(data)
            loss=calc_loss(output_cls, label)
            if (batch_id+1)%50==0:
                print("batch_id: ",batch_id)
            all_seriesname.extend(seriesname.data.cpu().numpy().tolist())
            all_predict_cls_1.extend(output_cls.data.cpu().numpy()[:,0].tolist())
            all_true_cls_1.extend(label.data.cpu().numpy()[:,0].tolist())
            all_loss_cls.append(loss.item())

        val_loss = np.array(all_loss_cls).sum() / len(all_loss_cls)
        thresholds_dic={0:th,1:th2,2:th3}
        predict_label_list=[[all_true_cls_1,all_predict_cls_1 ]]
        for k,(all_true_cls,all_predict_cls) in enumerate(predict_label_list):
            all_seriesname_tmp=np.array(all_seriesname)
            indexs=np.arange(len(all_seriesname_tmp)).reshape(-1,1)
            val_acc = ACC(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
            val_auc, fprs, tprs, thresholds= AUC(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
            val_fn = fc_false_negative(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
            val_fp = fc_false_positive(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]))
            tp, tn, th_tmp = yuedenindex_m(fprs, tprs, thresholds)
            if filename == "train":
                thresholds_dic[k]= th_tmp if 0.01 <= th_tmp <= 0.99 else  thresholds_dic[k]

            val_acc_th = ACC(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]), th=thresholds_dic[k])
            val_fn_th = fc_false_negative(y_true=list(np.array(all_true_cls)[indexs]),y_pred=list(np.array(all_predict_cls)[indexs]), th=thresholds_dic[k])
            val_fp_th = fc_false_positive(y_true=list(np.array(all_true_cls)[indexs]), y_pred=list(np.array(all_predict_cls)[indexs]), th=thresholds_dic[k])
            result_tmp=[round(val_auc, 4), round(val_acc_th, 4), round(1 - val_fn_th, 4), round(1 - val_fp_th, 4),
                round((1 - val_fp_th) * (1 - val_fn_th), 4), round(abs((1 - val_fp_th) - (1 - val_fn_th)), 4),
                round(val_acc, 4), round(1 - val_fn, 4), round(1 - val_fp, 4), round((1 - val_fp) * (1 - val_fn), 4),round(abs((1 - val_fp) - (1 - val_fn)), 4)]
            result.extend(result_tmp)
        return result+[ val_loss], thresholds_dic[0]



if __name__=="__main__":
    print("start:..")
    date=arg.date
    csvpath=os.path.join(arg.csvpath,arg.version,)
    eventpath=os.path.join(arg.eventpath,arg.version,arg.samplestyle)
    weightpath=os.path.join(arg.weightpath,arg.version,arg.samplestyle)
    batch_size,samplefactor =arg.batch_size,arg.samplefactor
    num_classes=arg.num_classes
    gpu,lr,min_lr,weight_decay=arg.gpu,arg.lr,arg.min_lr,arg.weight_decay
    train(csvpath, eventpath, weightpath, batch_size, samplefactor, num_classes, gpu, lr, min_lr, weight_decay)
