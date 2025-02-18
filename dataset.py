import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class ImageDataset(Dataset):
    def __init__(self, data, img_size=512, train=True, random_ratio=False):
        self.data = data
        self.img_size = img_size
        self.train = train
        self.random_ratio = random_ratio

    def __len__(self):
        if self.train:
          return len(self.data['train'])
        else:
          return len(self.data['test'])

    def __getitem__(self, idx):
        try:
            if self.train:
              idx_data = self.data['train'][idx]
            else:
              idx_data = self.data['test'][idx]
            img = idx_data['image']
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
            img = image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            prompt = idx_data['text']
            return img, prompt
        except Exception as e:
            print(e)


def loader(train_batch_size, num_workers, data):
    dataset = ImageDataset(data)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)