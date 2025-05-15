import numpy as np
from torchvision import transforms
from PIL import Image
import numpy as np
import torch


def read_image(img_path, resize_size=256, crop_size=224):

    img = Image.open(img_path).convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),  # TODO
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    img = transform_test(img)

    return img


path = 'D://njupt//新能源//transformer-fault-prediction-with-99-auc-main//transformer-fault-prediction-with-99-auc-main//images//transformer_no.png'

a = read_image(path)

print("========================")
