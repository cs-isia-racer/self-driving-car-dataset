from os import path

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

root = "./dataset/driving_dataset"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Pipeline:
    def __init__(self, feat_model, model):
        self.feat_model = feat_model
        self.model = model
        
    def predict(self, images, process=True):
        if process:
            images = torch.stack([preprocess(img) for img in images])
        
        with torch.no_grad():
            features = self.feat_model(images)
            
        return self.model.predict(features)
    
def stripped_resnet18():
    feat_model = models.resnet18(pretrained=True)

    # Strip the last linear layer
    feat_model.fc = nn.Sequential()
    return feat_model

def load_img(img_file, process=True):
    img = Image.open(path.join(root, img_file))
    if process:
        img = preprocess(img)
    return img

def extract_features(model, loader, verbose=True):
    X = []
    y = []
    
    count = 0
    with torch.no_grad():
        for (data, out) in loader:
            print('.', end='')
            X.append(model(data))
            y.extend(out)
            count += 1
            
            if count % 10 == 0:
                print()
            
    return np.vstack(X), np.array(y)
    
def load(path):
    return pd.read_csv(path, sep=' ', names=['image', 'steering'])


# This will implement the Dataset class for our Image Dataset
class ImageDataset(Dataset):
    def __init__(self, df, limit=None):
        self.df = df

        if limit is not None:
            self.df = self.df.iloc[:limit]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = load_img(self.df.iloc[idx, 0])
        feats = self.df.iloc[idx, 2]

        return (image, feats)

def create_loader(limit, workers=4, batch_size=256, shuffle=True):
    df = load(path.join(root, "data.txt"))
    scaler = MinMaxScaler()
    df['steering_clipped'] = df['steering'].clip(-135, 135)
    df['steering_scaled'] = scaler.fit_transform(df['steering_clipped'].values.reshape(-1, 1))
    
    dataset = ImageDataset(df, limit=limit)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=workers)
    return (df, loader, scaler)