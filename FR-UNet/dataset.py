import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation
from PIL import Image
import numpy as np

class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None, return_thin_vessels_mask = False):

        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        self.return_thin_vessels_mask = return_thin_vessels_mask



    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        gt_file = "gt" + img_file[3:]
        # if idx == 21606:
        #     print(img_file) 
        #     print(gt_file)
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()
        if self.return_thin_vessels_mask:
            w_file = "w" + img_file[3:]
            with open(file=os.path.join(self.data_path, w_file), mode='rb') as file:
                w = torch.from_numpy(pickle.load(file)).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            comb = torch.cat([img,gt],dim=0)
            
            # Return weight mask for the gt alongside the img and gt
            if self.return_thin_vessels_mask:
                comb = torch.cat([w,comb]) #[w,img,gt]
                comb = self.transforms(comb)
                return torch.unsqueeze(comb[0],0), torch.unsqueeze(comb[1],0), torch.unsqueeze(comb[2],0)
            else:
                comb = self.transforms(comb)
                return torch.unsqueeze(comb[0],0), torch.unsqueeze(comb[1],0)
        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)
    

def main():
    # UNIT TESTS
    # Generate weight masks for test dataset (samples for the Undergraduate Thesis presentation)
    dp = "/home/joaolinaris/USP/IC/Projeto_retina/Datasets/SAMPLE_CHASEDB1"
    ds = vessel_dataset(dp, mode="training", return_thin_vessels_mask=True)
    print(len(ds))
    for i,(w,img,gt) in enumerate(ds):

        if i == 21606:
            print(w.shape)
            print(img.shape)
            print(gt.shape)

            # img is normalized -> unrestorable: added an if statement to print the file path

            # img = Image.fromarray((255*img[0]).numpy().astype(np.uint8))
            # img.save("img_patch_sample.png")
            # gt = Image.fromarray((255*gt[0]).numpy().astype(np.uint8))
            # gt.save("gt_patch_sample.png")
            # w = w.numpy()[0]
            # w = (255*(w-w.min())/(w.max()-w.min())).astype(np.uint8)
            # w = Image.fromarray(w)
            # w.save("w_patch_sample.png")
            # print(torch.unique(w))
            # print(torch.unique(img))
            # print(torch.unique(gt))
            # all working great 




if __name__ == "__main__":
    main()
