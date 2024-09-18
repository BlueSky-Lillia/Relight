import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import cv2
import os
import random
import json
import numpy as np
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def image_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])

def center_crop(image, crop_size):
    
    height, width, _ = image.shape
    
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    return cropped_image

def Resize_512(iarray):
    iarray = iarray.astype(np.uint8)
    image = Image.fromarray(iarray)
    resized_image = image.resize((512, 512), Image.LANCZOS)
    resized_iarray = np.array(resized_image)
    return resized_iarray

def crop_image(image1, image2, image3):
    '''
    image1: Ground Truth
    image2: Mask
    image3: Decomposed Image
    return Ground Truth Background with decompsoed human region
    '''
    from PIL import Image, ImageOps

    bbox = image2.getbbox()

class RelightDataset(Dataset):
    def __init__(self, root=None):
        self.data = []
        with open(os.path.join(root, 'train.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.root = root # 路径
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_name = item['name']
        gt_path = item['gt_paths']
        mask_path =item['mask_paths']
        decompose_path = item['decompose_paths'] # decomposed background
        background_path = item['mask_paths'].replace('masks_360p', 'inpaint')
        albedo_path = item['albedo_paths'] # decomposed human after matting
        background_caption_path = item['background_caption_paths']
        gt_caption_path = item['gt_caption_paths']

        # Do not forget that OpenCV read images in BGR order.
        gt = cv2.imread(os.path.join(self.root, gt_path.replace('./', '')))
        mask = cv2.imread(os.path.join(self.root, mask_path.replace('./', '')))
        decompose = cv2.imread(os.path.join(self.root, decompose_path.replace('./', '').replace('shading', 'albedo'))) 
        albedo = cv2.imread(os.path.join(self.root, albedo_path.replace('./', '').replace('shading', 'albedo').replace('decompose', 'p_decompose'))) 
        background = cv2.imread(os.path.join(self.root, background_path.replace('./', '')))

        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        decompose = cv2.cvtColor(decompose, cv2.COLOR_BGR2RGB)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        # Resize to the same 
        h1, w1, _ = gt.shape
        h2, w2, _ = mask.shape
        h3, w3, _ = decompose.shape
        h4, w4, _ = albedo.shape
        h5, w5, _ = background.shape

        # 选取所有图片中的最小宽度和高度
        min_dim = min(h1, w1, h2, w2, h3, w3, h4, w4)

        # 对每张图片进行中心裁剪为min_dim x min_dim的正方形
        gt = center_crop(gt, min_dim)
        mask = center_crop(mask, min_dim)
        decompose = center_crop(decompose, min_dim)
        albedo = center_crop(albedo, min_dim)
        background = center_crop(background, min_dim)

        gt = Resize_512(gt)
        mask = Resize_512(mask)
        decompose = Resize_512(decompose)
        albedo = Resize_512(albedo)
        background = Resize_512(background)

        gt = Image.fromarray(gt)
        mask = Image.fromarray(mask).convert("L")
        decompose = Image.fromarray(decompose)
        albedo = Image.fromarray(albedo)
        background = Image.fromarray(background)
        composition = gt.copy()
        
        composition.paste(albedo, (0, 0), mask)
        gt = np.array(gt)
        mask = np.array(mask.convert("RGB"))
        composition = np.array(composition)
        albedo = np.array(albedo)
        decompose = np.array(decompose)
        background = np.array(background)

        # source = np.concatenate((decompose, albedo, mask), axis=-1)

        # Normalize source images to [0, 1] used to be the control condition, which means transform the pixel value of unit8（0-255） to floate (0.0 - 1.0)
        mask = mask.astype(np.float32) / 255.0
        decompose = decompose.astype(np.float32) / 255.0
        albedo = albedo.astype(np.float32) / 255.0
        background = background.astype(np.float32) / 255.0
        composition = composition.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1] used to generate image.
        gt = (gt.astype(np.float32) / 127.5) - 1.0

        # String
        with open(os.path.join(self.root, background_caption_path), 'r', encoding='utf-8') as bg_caption:
            data = json.load(bg_caption)
            background_caption = data[0]
        
        with open(os.path.join(self.root, gt_caption_path), 'r', encoding='utf-8') as gt_caption:
            data = json.load(gt_caption)
            gt_caption = data[0]

        lighting_caption = {}
        lighting_caption['Background'] = background_caption
        lighting_caption['Ground Truth'] = gt_caption
        caption = lighting_caption['Ground Truth']
        prompt = 'relighting portrait'
        
        return dict(jpg=gt, txt=prompt, hint=composition, caption=caption, composition=composition, mask=mask, lighting=lighting_caption)
    
    def display(self):
        random_index = random.randint(0, len(self.data)-1)
        item = self.data[random_index]
        image_name = item['name']
        gt_path = item['gt_paths']
        mask_path =item['mask_paths']
        decompose_path = item['decompose_paths'] # decomposed background
        albedo_path = item['albedo_paths'] # decomposed human after matting
        background_caption_path = item['background_caption_paths']
        gt_caption_path = item['gt_caption_paths']

        gt = cv2.imread(os.path.join(self.root, gt_path.replace('./', '')))
        # print(os.path.join(self.root, gt_path.replace('./', '')))
        mask = cv2.imread(os.path.join(self.root, mask_path.replace('./', '')))
        decompose = cv2.imread(os.path.join(self.root, decompose_path.replace('./', '').replace('shading', 'albedo'))) 
        albedo = cv2.imread(os.path.join(self.root, albedo_path.replace('./', '').replace('shading', 'albedo').replace('decompose', 'p_decompose'))) 

        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # print('mask.shape:', mask.shape)
        decompose = cv2.cvtColor(decompose, cv2.COLOR_BGR2RGB)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        with open(os.path.join(self.root, background_caption_path), 'r', encoding='utf-8') as bg_caption:
            data = json.load(bg_caption)
            background_caption = data[0]
        
        with open(os.path.join(self.root, gt_caption_path), 'r', encoding='utf-8') as gt_caption:
            data = json.load(gt_caption)
            gt_caption = data[0]

        # 获取每张图片的尺寸
        h1, w1, _ = gt.shape
        h2, w2, _ = mask.shape
        h3, w3, _ = decompose.shape
        h4, w4, _ = albedo.shape

        # 选取所有图片中的最小宽度和高度
        min_dim = min(h1, w1, h2, w2, h3, w3, h4, w4)

        # 对每张图片进行中心裁剪为min_dim x min_dim的正方形
        image1 = center_crop(gt, min_dim)
        image2 = center_crop(mask, min_dim)
        image3 = center_crop(decompose, min_dim)
        image4 = center_crop(albedo, min_dim)

        image1 = Resize_512(image1)
        image2 = Resize_512(image2)
        image3 = Resize_512(image3)
        image4 = Resize_512(image4)

        top_row = np.concatenate((image1, image2), axis=1)  # axis=1表示水平拼接
        bottom_row = np.concatenate((image3, image4), axis=1)
        combined_image = np.concatenate((top_row, bottom_row), axis=0)  # axis=0表示垂直拼接

        # 保存最终合并的图片
        cv2.imwrite(f'{random_index}.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    def display2(self, random_index=None):
        if random_index is None:
            random_index = random.randint(0, len(self.data)-1)
        item = self.data[random_index]
        image_name = item['name']
        gt_path = item['gt_paths']
        mask_path =item['mask_paths']
        decompose_path = item['decompose_paths'] # decomposed background
        background_path = item['mask_paths'].replace('masks_360p', 'inpaint')
        albedo_path = item['albedo_paths'] # decomposed human after matting
        background_caption_path = item['background_caption_paths']
        gt_caption_path = item['gt_caption_paths']

        # Do not forget that OpenCV read images in BGR order.
        gt = cv2.imread(os.path.join(self.root, gt_path.replace('./', '')))
        mask = cv2.imread(os.path.join(self.root, mask_path.replace('./', '')))
        decompose = cv2.imread(os.path.join(self.root, decompose_path.replace('./', '').replace('shading', 'albedo'))) 
        albedo = cv2.imread(os.path.join(self.root, albedo_path.replace('./', '').replace('shading', 'albedo').replace('decompose', 'p_decompose'))) 
        background = cv2.imread(os.path.join(self.root, background_path.replace('./', '')))

        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        decompose = cv2.cvtColor(decompose, cv2.COLOR_BGR2RGB)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        with open(os.path.join(self.root, background_caption_path), 'r', encoding='utf-8') as bg_caption:
            data = json.load(bg_caption)
            background_caption = data[0]
        
        with open(os.path.join(self.root, gt_caption_path), 'r', encoding='utf-8') as gt_caption:
            data = json.load(gt_caption)
            gt_caption = data[0]

        # 获取每张图片的尺寸
        h1, w1, _ = gt.shape
        h2, w2, _ = mask.shape
        h3, w3, _ = decompose.shape
        h4, w4, _ = albedo.shape
        h5, w5, _ = background.shape

        # 选取所有图片中的最小宽度和高度
        min_dim = min(h1, w1, h2, w2, h3, w3, h4, w4)

        # 对每张图片进行中心裁剪为min_dim x min_dim的正方形
        image1 = center_crop(gt, min_dim)
        image2 = center_crop(mask, min_dim)
        image3 = center_crop(decompose, min_dim)
        image4 = center_crop(albedo, min_dim)
        image5 = center_crop(background, min_dim)

        image1 = Resize_512(image1)
        image2 = Resize_512(image2)
        image3 = Resize_512(image3)
        image4 = Resize_512(image4)
        image5 = Resize_512(image5)
        print(image1.shape, image2.shape, image3.shape)

        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2).convert("L")
        image3 = Image.fromarray(image3)
        image4 = Image.fromarray(image4)
        image6 = image1.copy()
        
        image6.paste(image4, (0, 0), image2)
        image2 = np.array(image2.convert("RGB"))
        image3 = np.array(image6)
        image4 = np.array(image4)

        top_row = np.concatenate((image1, image2), axis=1)  # axis=1表示水平拼接
        bottom_row = np.concatenate((image6, image5), axis=1)
        combined_image = np.concatenate((top_row, bottom_row), axis=0)  # axis=0表示垂直拼接

        # 保存最终合并的图片
        cv2.imwrite(f'{random_index}.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

if __name__ =='__main__':
    dataset = RelightDataset(root = '/home/wangzhen/Data/Raw/ppr10k')
    dataset.display2()