import os
import scipy.stats
from data import BBDataset
from torch.utils.data import DataLoader
from models.model import SAAN
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from common import *
import argparse
import torch.nn.functional as F
import pandas as pd
import scipy
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
import cv2 as cv


mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]

transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='run_images')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/BAID')
    parser.add_argument('--checkpoint_name', type=str,
                        default='model_best.pth')

    return parser.parse_args()


def test(args):
    device = args.device
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = SAAN(num_classes=1)
    model = model.to(device)
    if device == torch.device('cuda'):
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    image_dir = args.directory

    with torch.no_grad():
        for filename in os.listdir(image_dir):
            # Load the image
            image_path = os.path.join(image_dir, filename)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # Apply transformations
            image_tensor = transform(image).to(device) 

            image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension
            # Perform inference
            with torch.no_grad():
                predicted_label = model(image_tensor)
                prediction = predicted_label.squeeze().cpu().numpy() * 10
                print(filename, prediction)




if __name__ == '__main__':
    args = parse_args()
    test(args)
