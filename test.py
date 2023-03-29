import pandas as pd
import numpy as np
import os
import glob
import torch
import cv2
import tqdm
import random
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms  import functional as F
from tensorboardX import SummaryWriter
from PIL import Image
import argparse
from data_process import Resize,RandomCropChannel,RandomHorizontalFlip,RandomVerticalFlip,Normalization,ToTensor
from data_process import MRDataset,Compose


from model import Transformer_2d
from metrics import AUC,ACC,fc_false_positive,fc_false_negative
import time
from torchvision.models._utils import IntermediateLayerGetter
ap = argparse.ArgumentParser()  # cell_detect
ap.add_argument("--csvpath", default="./RAIN/csv", type=str)
ap.add_argument("--outpath", default="./RAIN/output", type=str)
ap.add_argument("--weightpath", default="./RAIN/weights", type=str)
ap.add_argument("--version", default="vs12",type=str)
ap.add_argument("--date", default="y-m-d",type=str)
ap.add_argument("--gpu", default="0", type=str)
ap.add_argument("--lr", default=1e-4, type=float)
ap.add_argument("--min_lr", default=1e-5, type=float)
ap.add_argument("--weight_decay", default=0, type=float)
ap.add_argument("--epochs",type=str)
ap.add_argument("--batch_size", default=8, type=int)
ap.add_argument("--samplefactor", default=1.0, type=float)
ap.add_argument("--num_classes",default=2,type=int)
ap.add_argument("--subsample",default=0,type=int)
ap.add_argument("--samplestyle",default="None", type=str)
ap.add_argument("--port", default="23456", type=str)
ap.add_argument("--num_workers",default=12,type=int)#12
ap.add_argument("--dcmnum", default=16,type=int)
ap.add_argument("--backbone_name",default="resnet18", type=str)
ap.add_argument("--image_size",default=448, type=int)
ap.add_argument("--d_model", default=256, type=int)
ap.add_argument("--tf_n_layers", default=4, type=int)
ap.add_argument("--tf_n_head", default=6, type=int)


arg = ap.parse_args()
"""自定义transforms.Compose"""


test_transform=Compose(transforms=[Resize((arg.image_size,arg.image_size)),  RandomCropChannel(length=arg.dcmnum,dataset="test") ,
                                   Normalization(),ToTensor()])
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu

def collate_fn(batch_list):
    data = torch.cat([i[0].unsqueeze(0) for i in batch_list], 0)
    label = torch.cat([i[1] for i in batch_list], 0).unsqueeze(1)
    seriesname = torch.cat([i[2] for i in batch_list], 0).unsqueeze(1)
    return data, label, seriesname


def my_DataLoader(csvpath,batch_size, samplefactor = 1,dataset="none"):
    test_dataset = MRDataset(csvpath=csvpath, dataset=dataset, transforms = test_transform, samplefactor = 1,subsample=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
                                num_workers=18,collate_fn=collate_fn)
    return test_dataloader


def selftest(csvpath,eventpath,weightpath,batch_size, samplefactor = 1,num_classes=2,gpu="3",lr=1e-4,min_lr=5e-5,weight_decay=0.0):
    Model = Transformer_2d(backbone_name=arg.backbone_name, image_size=arg.image_size, num_classes=num_classes,
                    total_channels=arg.dcmnum, unit_channel=arg.unit_channel, d_model=arg.d_model, d_inner=4 * arg.d_model, n_layers=arg.tf_n_layers,
                    n_head=arg.tf_n_head, d_k=64, d_v=64,dropout = 0.1,pretrained = 1,backbone_feature_dim = None, pool = "cls", feature_patch_d = 2, feature_patch_h = 1)

    epoch=arg.epochs
    flags = "T1C"
    currentweights_tmp = glob.glob(os.path.join(weightpath, str(epoch) + "_train_*"))
    print(os.path.join(weightpath, str(epoch) + "_train_*"))
    weightpath_tmp=currentweights_tmp[0]
    print("currentweight :", weightpath_tmp)
    print("load weights...start")
    stat_dict = torch.load(weightpath_tmp)
    Model.load_state_dict(stat_dict)
    outpath_tmp = os.path.join(arg.outpath, "result",arg.version)
    if not os.path.exists(outpath_tmp):
        os.makedirs(outpath_tmp)
    writer = pd.ExcelWriter(os.path.join(outpath_tmp, "_".join([arg.date, flags, str(epoch), "predict.xlsx"])))
    print("load weights...end")
    Model.cuda()
    print("arg.externalset:",arg.externalset)
    for dataset in ["test"]:
        tmp_dataloader =my_DataLoader(csvpath, batch_size, samplefactor, dataset=dataset)
        Data=val(Model=Model, dataloader=tmp_dataloader,dataset=dataset,outpath=arg.outpath)
        Data.to_excel(writer,sheet_name=dataset)
    writer.close()


def val(Model,dataloader,dataset,outpath):
    Model.eval()
    all_true_cls,all_predict_cls = [], []
    all_seriesname=[]

    flags ="T1C"
    with torch.no_grad():
        for batch_id, (data,  label, seriesname) in tqdm.tqdm(enumerate(dataloader)):
            data, label, seriesname = data.float().cuda(), label.float().cuda(), seriesname.float().cuda()
            data=data.squeeze(dim=1)
            output_cls = Model(data)
            all_predict_cls.extend(output_cls.data.cpu().numpy().tolist())
            all_true_cls.extend(label.data.cpu().numpy().tolist())
            all_seriesname.extend(seriesname.data.cpu().numpy().tolist())
            if batch_id%50==0:
                print("val_batch_id:",batch_id)

        test_acc = ACC(y_true=all_true_cls, y_pred=all_predict_cls)
        # test_loss = np.array(all_loss).sum() / len(all_loss)
        test_auc,_,_,_= AUC(y_true=all_true_cls, y_pred=all_predict_cls)
        test_fn = fc_false_negative(y_true=all_true_cls, y_pred=all_predict_cls)
        test_fp = fc_false_positive(y_true=all_true_cls, y_pred=all_predict_cls)
        print("*_"*20,dataset,":\n")
        print("test_auc: {:.4f}, test_acc:  {:.4f},1-test_fn:  {:.4f},1-test_fp: {:.4f}".format(test_auc ,test_acc,1-test_fn,1-test_fp))

        dicts = { "label":[int(i[0]) for i in all_true_cls],
          "seriesname":[int(i[0]) for i in all_seriesname],
          "predict":[round(i[0], 4) for i in all_predict_cls]}
        data = pd.DataFrame(dicts)
        print("data_length::", len(data))
        return data.loc[:, ["label", "seriesname", "predict"]]



def save_excel(data,outpath,samplename):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outpath_tmp = os.path.join(outpath, samplename)
    data.to_excel(outpath_tmp,index=False)

def save_npy(array_t,outpath,samplename):
    array=(array_t>=0.5).astype(np.int8)## 8位存储
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outpath_tmp=os.path.join(outpath,samplename)
    np.save(outpath_tmp,array)

if __name__=="__main__":
    print("start:..")
    csvpath=os.path.join(arg.csvpath,arg.version)
    eventpath=arg.eventpath
    weightpath=arg.weightpath
    batch_size,samplefactor =arg.batch_size,arg.samplefactor
    num_classes=arg.num_classes
    gpu,lr,min_lr,weight_decay=arg.gpu,arg.lr,arg.min_lr,arg.weight_decay

    selftest(csvpath, eventpath, weightpath, batch_size, samplefactor, num_classes, gpu, lr, min_lr, weight_decay, )
