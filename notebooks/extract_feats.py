import pandas as pd
import os
from os import path
import torch
from torch import nn
from torchvision import transforms, models
import PIL
from tqdm.auto import tqdm
import numpy as np
from logging import info

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

info('Read csv')
df_train = pd.read_csv('/cvhci/data/praktikumSS2019/gruppe1/google-landmark/valid_train.csv')


arch = 'resnet50'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)


model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

model = nn.Sequential(*(list(model.children())[:-1]))

model.eval()
model = model.to('cuda')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.paths = df["path"].to_numpy()
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: PIL.Image.open(x).convert("RGB")),
            transforms.RandomCrop(size=(224,224), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, idx):
        return self.transforms(self.paths[idx])
        
    def __len__(self):
        return len(self.paths)

ds = Dataset(df_train)
bs = 256
dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs, num_workers=8, shuffle=False)

results = torch.empty(len(ds), 2048)
with torch.no_grad():
    for idx, img in enumerate(tqdm(dl)):
        result = model(img.cuda()).view(-1, 2048)
        results[bs*idx:bs*(idx+1)] = result
model.cpu()

info('save feats')
np.save('/cvhci/data/praktikumSS2019/gruppe1/google-landmark/valid_train_wsl_feats.npy', results)
info('finished')
