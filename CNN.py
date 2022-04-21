import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import numpy as np

transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()
])
path = 'E:/ECE884-489286/ECE884PL/dataimg/0.jpg' # 要转换为图片的.npy文件
imageload = cv2.imread(path)


vggnet16 = models.vgg16(pretrained=True)
# Transform the image
imag = transform(imageload)
# Reshape the image. PyTorch model reads 4-dimensional tensor
# [batch_size, channels, width, height]
img = imag.reshape(1, 3, 448, 448)
# img = img.to(device)
# We only extract features, so we don't need gradient
with torch.no_grad():
    # Extract the feature from the image
    feature = vggnet16(img)
    ff = feature.numpy()

# Convert to NumPy Array, Reshape it, and save it to features variable
# features.append(feature.cpu().detach().numpy().reshape(-1))

