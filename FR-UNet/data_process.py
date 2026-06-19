import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor, ToPILImage
from utils.helpers import dir_exists, remove_files
from retinal_thin_vessels.weights import get_weight_mask

def data_process(data_path, name, patch_size, stride, mode, write_png=False, compute_gt_weights=False):
    save_path = os.path.join(data_path, f"{mode}_pro")
    save_path_png = ""
    if write_png:
        save_path_png = os.path.join(data_path, f"{mode}_pro_png")
        dir_exists(save_path_png)
        remove_files(save_path_png)
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        img_path = os.path.join(data_path, "input")
        gt_path = os.path.join(data_path, "target")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    w_list = []
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

        elif name == "CHASEDB1":
            if len(file) >= 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(img_path, file))
                    gt = Image.open(os.path.join(
                        gt_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(img_path, file))
                    gt = Image.open(os.path.join(
                        gt_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        
        if mode == "training" and compute_gt_weights:
            # Compute and save weights for gt (arrayOfW[N,1,H,W])
            w_list.append(torch.from_numpy(get_weight_mask(ToTensor()(gt),weights_function=1)))

    img_list = normalization(img_list)
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)     
        save_patch(img_patch, save_path, "img_patch", name, write_png, save_path_png) #[N,1,H,W]
        save_patch(gt_patch, save_path, "gt_patch", name, write_png, save_path_png) #[N,1,H,W]
        if compute_gt_weights: 
            w_patch = get_patch(w_list,patch_size,stride)
            save_patch(w_patch, save_path, "w_patch", name) #[N,1,H,W] (extra channel facilitates loss computation)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name,write_png, save_path_png)
        save_each_image(gt_list, save_path, "gt", name,write_png, save_path_png)


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


def save_patch(imgs_list, path, type, name, write_png=False, save_path_png=""):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            # if "w_patch" in type:
            #     sub = sub[0] #[1,H,W] -> [H,W]
            pickle.dump(sub.numpy(), file)
            # print(f'save {name} {type} : {type}_{i}.pkl')
        
        # Optionally writes the file in png format
        if write_png:
            # Creates a separate png dir
            with open(file=os.path.join(save_path_png, f'{type}_{i}.png'), mode='wb') as file:
                pilImg = ToPILImage()(sub)
                pilImg.save(file)

    print(f'save patch completed! saved {len(imgs_list)} imgs')

# def save_w_patch(w_list, path, type, name):
#     # Saves the patches in w_list in numpy format

def save_each_image(imgs_list, path, type, name, write_png=False, save_path_png=""):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')

        # Optionally writes the file in png format
        if write_png:
            # Creates a separate png dir
            with open(file=os.path.join(save_path_png, f'{type}_{i}.png'), mode='wb') as file:
                pilImg = ToPILImage()(sub)
                pilImg.save(file)


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/DRIVE", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','CHUAC','DCA1'],required=True)
    parser.add_argument('-wp', '--write_png', default="false", type=str,
                        help='boolean (true or false) indicating if must write to png file or not')
    parser.add_argument('-ps', '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=6,
                        help='the stride of image partition')
    parser.add_argument('-cw', '--compute_weights', default="false", type=str,
                        help='flag for computing weights based on vessels thickness')
    args = parser.parse_args()
    with open('config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  # 为列表类型

    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "training", args.write_png == "true", args.compute_weights == "true")
    # data_process(args.dataset_path, args.dataset_name,
    #              args.patch_size, args.stride, "test", args.write_png == "true")
