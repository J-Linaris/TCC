import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation
from retinal_thin_vessels.weights import get_weight_mask


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
        if return_thin_vessels_mask:
            # Compute weight masks for all gts
            # self.w_array = []
            # for _, file in enumerate(self.img_file):
            #     file = open(file=os.path.join(self.data_path, "gt" + file[3:]), mode='rb')
            #     gt = torch.from_numpy(pickle.load(file)).float()
            #     self.w_array.append(torch.from_numpy(get_weight_mask(gt,1)))
            
            # JUST EXPERIMENTING WITH THE DIMENSIONS
            file = open(file=os.path.join(self.data_path, "gt" + self.img_file[0][3:]), mode='rb')
            gt = torch.from_numpy(pickle.load(file)).float()
            w = torch.from_numpy(get_weight_mask(gt,1))
            print(gt.size())
            print(w.size())



    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

            # Return weight mask for the gt alongside the img and gt
            if self.return_thin_vessels_mask:
                w = self.w_array[idx]
                return w, img, gt
        
        # Check the dimension returned by img and gt for us to set what we expect the shape of the loss to be (will be the same as the gt shape (and pred))
        # Probably [1,H,W]
        print(img.shape) # --> execute once in main() and delete
        print(gt.shape)
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
    # Generate weight masks for test dataset
    dp = "/home/joaolinaris/USP/IC/Projeto_retina/Datasets/CHASEDB1"
    ds = vessel_dataset(dp, mode="test", return_thin_vessels_mask=True)
    # a = next(iter(ds))
    # print(a)

if __name__ == "__main__":
    main()
