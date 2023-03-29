import pandas as pd
import numpy as np
import os
import glob
import torch
import cv2
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms  import functional as F
from PIL import Image
import SimpleITK as sitk
import torchio as tio
import torchio
import SimpleITK  as sitk



"""自定义transforms.Compose"""
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data,img1,img2):
        for t in self.transforms:
            data,img1,img2= t(data,img1,img2)
        return data,img1,img2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomAffine(object):
    def __init__(self,    scales=(0.8,1.2,0.8,1.2,1,1),degrees=(0,0,20),translation=(10,10,0),center='image',
            default_pad_value='minimum',image_interpolation='linear',label_interpolation='nearest',check_shape=True,p=0.5):
        self.ramdomaffine= torchio.transforms.RandomAffine(scales=scales,degrees = degrees,translation = translation ,
                  center = center,default_pad_value =default_pad_value,image_interpolation = image_interpolation,
                    label_interpolation = label_interpolation,check_shape=check_shape)
        self.p=p
    def __call__(self,img,img1,img2):
        if self.p > random.random():
            return self.ramdomaffine(img),img1,img2
        return img,img1,img2


class Swap_img_array(object):
    def __ini__(self):
        pass
    def __call__(self, input,img1,img2):
        if isinstance(input,sitk.SimpleITK.Image):
            output = sitk.GetArrayFromImage(input)## z,y,x
            output = np.transpose(output, axes=[1, 2, 0])## y,x,z
        elif isinstance(input, np.ndarray):
            output = sitk.GetImageFromArray(np.transpose(input, axes=[2, 0, 1]))
            output.SetSpacing((0.5, 0.5, 6))
        return output,img1,img2


class ToTensor(object):
    def __call__(self, ndarray,img1,img2):
        ndarray= torch.from_numpy(np.expand_dims(np.transpose(ndarray,axes=(2, 0, 1)),axis=0).copy())#:math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`,
        img1 = torch.from_numpy(np.expand_dims(np.transpose(img1, axes=(2, 0, 1)),
                                                  axis=0).copy())  #:math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`,
        img2 = torch.from_numpy(np.expand_dims(np.transpose(img2, axes=(2, 0, 1)),
                                                  axis=0).copy())  #:math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`,
        return ndarray,img1,img2


class Normalization(object):
    def __init__(self):
         pass

    def __call__(self, ndarray,img1,img2):
        img1 = ndarray[:, :,1::3]
        img2 = ndarray[:, :, 2::3]
        ndarray = ndarray[:, :, 0::3]  ## 还原
        ndarray=self.normalize(ndarray)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return ndarray,img1,img2##

    def normalize(self,ndarray):
        ndarray=self.convert_to_2e12(ndarray)
        ndarray = ndarray / (ndarray[ndarray > ndarray.mean()].mean())
        return ndarray

    def convert_to_2e12(self,ndarray):  #
        ndarray = ndarray.astype(np.float64)
        ndarray = (ndarray  / ndarray .max()) * np.power(2, 12)
        return ndarray

class Resize(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR,p=0.1):
        self.size = size  ##(w,h)
        self.interpolation = interpolation
        self.dataset=dataset

    def __call__(self, ndarray,img1,img2):
        """
        Args:
            ndarray:(h,w,c)
        """
        ndarray=cv2.resize(ndarray.astype(np.float32), self.size, self.interpolation)
        img1 = cv2.resize(img1.astype(np.float32), self.size, self.interpolation)
        img2= cv2.resize( img2.astype(np.float32), self.size, self.interpolation)
        return ndarray,img1,img2

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
      Args:
          p (float): probability of the image being flipped. Default value is 0.5
      """

    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, ndarray,img1, img2):
        """
        Args:
            img (PIL Image): ndarrayto be flipped.
        Returns:
            PIL Image: Randomly flipped ndarray.
        """
        if random.random() < self.p:
            return ndarray[:,::-1,:],img1, img2
        return ndarray,img1, img2

class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,ndarray,img1, img2):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return ndarray[::-1,:,:],img1,img2
        return ndarray,img1, img2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCropChannel(object):
    def __init__(self,length=30,dataset="test"):
        self.length=length
        self.p=p
        self.dataset=dataset

    def __call__(self,ndarray,img1,img2):
        slicenum=ndarray.shape[-1]
        if slicenum <=self.length:
            ndarray=self.fill_channel_with_zero(ndarray,slicenum)
            img1 = self.fill_channel_with_zero(img1, slicenum)
            img2 = self.fill_channel_with_zero(img2, slicenum)
            merge=np.concatenate([ ndarray,img1,img2],axis=-1)
        else:
            sindex, eindex=self.call_extend(ndarray)##
            merge=np.concatenate([ ndarray[:,:, sindex: eindex + 1],img1[:,:, sindex: eindex + 1],img2[:,:, sindex: eindex + 1]],axis=-1)

        for  i in range( self.length):
            if i==0:
                merge_sort=merge[:,:,i::self.length]
            else:
                merge_sort=np.concatenate([merge_sort,merge[:,:,i::self.length]],axis=-1)
        return   merge_sort,None,None#

    def call_extend(self,ndarray):
        slicenum=ndarray.shape[-1]
        if (slicenum - 3) > self.length:
            eindex = slicenum - 1 - 3
            sindex = eindex - self.length + 1
        else:
            medindex = slicenum // 2
            sindex = medindex - self.length // 2
            eindex = medindex + self.length // 2 - 1
        return sindex, eindex

    def fill_channel_with_zero(self,ndarray,slicenum):
        neednum=self.length-slicenum
        pad_s=neednum
        ndarray=np.pad(ndarray,((0,0),(0,0),(pad_s,0)),mode='constant')
        return ndarray


class MRDataset(Dataset):
    def __init__(self,csvpath,dataset="train",transforms=None,samplefactor=1,subsample=0,samplestyle="2*min"):
        self.csvpath=csvpath
        self.dataset=dataset
        self.transforms=transforms
        self.samplefactor = samplefactor
        self.subsample=usubsample
        self.samplestyle=samplestyle

        self.data_list,self.data_pre1_list,self.data_pre2_list,self.label_list,self.label_pre1_list,self.label_pre2_list,self.seriesname_list,self.T_list,self.P_list,self.L_list=self.read_csv()

    def __getitem__(self, index):
        data_pre0, data_pre1, data_pre2,label_pre0,label_pre1,label_pre2,T,P,L,samplepath=self.load_npy(index)
        seriesname_tmp = np.array([self.seriesname_list[index]]).astype(np.float32)

        if self.transforms is not None:
            data_pre0, data_pre1, data_pre2=self.transforms(data_pre0,data_pre1,data_pre2)
        data=torch.cat([data_pre0, data_pre1, data_pre2],dim=0)
        seriesname= torch.from_numpy(seriesname_tmp)
        label_pre0 = torch.from_numpy(np.array([label_pre0]).reshape(-1, 1)).float()
        label_pre1 = torch.from_numpy(np.array([label_pre1]).reshape(-1, 1)).float()
        label_pre2 = torch.from_numpy(np.array([label_pre2]).reshape(-1, 1)).float()

        return data.float(),label_pre0.squeeze(0),seriesname

    def load_npy(self,index):
        datapath_pre0=self.data_list[index]
        data_pre0=np.load(datapath_pre0)
        datapath_pre1 = self.data_pre1_list[index]
        data_pre1 = np.load(datapath_pre1)
        datapath_pre2= self.data_pre2_list[index]
        data_pre2 = np.load(datapath_pre2)
        data_pre0[data_pre0>65536]=0
        data_pre1[data_pre1 > 65536] = 0
        data_pre2[data_pre2> 65536] = 0

        label_pre0=self.label_list[index]
        label_pre1 = self.label_pre1_list[index]
        label_pre2 = self.label_pre2_list[index]
        T=self.T_list[index]
        P=self.P_list[index]
        L =self.L_list[index]
        return data_pre0,data_pre1,data_pre2,label_pre0,label_pre1,label_pre2,T,P,L,datapath_pre0

    def read_csv(self):

        paths = glob.glob(os.path.join(self.csvpath, self.dataset + ".*"))
        data_tmp=pd.read_csv(paths[0])
        data=data[data["seriesname"]==self.seriesflag].reset_index(drop=True)
        label_tmp = "label"
        classids = np.unique(data.loc[:, label_tmp].values).tolist()
        if self.dataset!="train" or (self.dataset=="train" and self.subsample==0) :
            return data.iloc[:,0].values.tolist(),data.iloc[:,1].values.tolist(),data.iloc[:,2].values.tolist(),data.iloc[:,3].values.tolist(),data.iloc[:,4].values.tolist(),data.iloc[:,5].values.tolist(), \
                   data.iloc[:, 6].values.tolist(), data.iloc[:, 7].values.tolist(),data.iloc[:, 8].values.tolist(), data.iloc[:, 9].values.tolist()
        if self.samplestyle=="max":
            classid_maxnum = max([len(data[data.loc[:, label_tmp] == id]) for id in classids])
        elif self.samplestyle == "min":
            classid_maxnum = min([len(data[data.loc[:, label_tmp] == id]) for id in classids])
        elif self.samplestyle == "2*min":
            classid_maxnum = 2*min([len(data[data.loc[:, label_tmp] == id]) for id in classids])
        else:
            assert self.samplestyle in ["max", "min","2*min"], print("self.samplestyle in [max,min，2*min]", self.samplestyle)
        data_merge = []
        for  id in classids:
            data_tmp = data[data.loc[:, label_tmp] == id].reset_index(drop=True)  #
            data_tmp=self.resample(classid_maxnum,data_tmp)
            data_merge.append(data_tmp)
        data_merge = pd.concat(data_merge).reset_index(drop=True)
        return data_merge.iloc[:,0].values.tolist(),data_merge.iloc[:,1].values.tolist(),data_merge.iloc[:,2].values.tolist(),\
               data_merge.iloc[:,3].values.tolist(),data_merge.iloc[:,4].values.tolist(),data_merge.iloc[:,5].values.tolist(), \
               data_merge.iloc[:, 6].values.tolist(), data_merge.iloc[:, 7].values.tolist(),data_merge.iloc[:, 8].values.tolist(), data_merge.iloc[:, 9].values.tolist()

    def resample(self,maxinum,data_L):## 支持降采样和重采样
        if maxinum<=len(data_L):
            return data_L.sample(n=maxinum,random_state=None) if  maxinum<len(data_L) else data_L
        expectnum = int(maxinum / self.samplefactor)
        tmp = data_L
        for i in range(1, expectnum // len(data_L)):
            tmp = pd.concat([tmp, data_L])
        if expectnum > len(data_L) and (expectnum % len(data_L) != 0):
            np.random.seed(seed=1)
            permuate = np.random.permutation(len(data_L))[:expectnum % len(data_L)]
            tmp = pd.concat([tmp, data_L.iloc[permuate, :]])
        return tmp

    def __len__(self):
        return len(self.data_list)