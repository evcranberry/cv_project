import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
import streamlit as st
import urllib.request

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_unet = smp.Unet(
    encoder_name="resnet101",  # Используем предобученный энкодер
    encoder_weights="imagenet",  # Веса, предобученные
    in_channels=3,  # 3 канала для RGB
    classes=1,  # 1 класс (бинарная сегментация)
    activation="sigmoid",  # Sigmoid для бинарной сегментации
)
weights_path = "models/model_weights_Unet_resnet101_1.pt"
model_unet.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model_unet.to(DEVICE)

model_segformer = smp.Segformer(
    encoder_name="resnet34",  
    encoder_weights="imagenet",  # Веса, предобученные на ImageNet
    in_channels=3,  # 3 канала для RGB
    classes=1,  # 1 класс (бинарная сегментация)
    activation="sigmoid",  # Sigmoid для бинарной сегментации
)
weights_path_seg = "models/model_weights_segform.pt"
model_segformer.load_state_dict(torch.load(weights_path_seg, map_location=torch.device("cpu")))
model_segformer.to(DEVICE)

IMAGE_SIZE = (256, 256)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

st.page_link('pages/Космос_метрики.py', label='Узнать детали обучения модели')
st.title('__Сегментируйте аэрокосмический снимок! 🗾__')
st.write('##### <- Загрузите снимок сюда')

uploaded_file = st.sidebar.file_uploader(label='Загружать снимок сюда:', type=['jpeg', 'png'], accept_multiple_files=True)
model_unet.eval()

model=None

st.write('Выберите модель:')
if st.button('Unet'):
    model = model_unet
if st.button('Segform'):
    model = model_segformer

if uploaded_file is not None:
    for file in uploaded_file:
        image = Image.open(file)
        st.write('Ваше изображение:')
        st.image(image)
        if model is not None:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            if model is not None:
                with torch.no_grad():
                    outputs = model(image_tensor)
                    pred_masks = (outputs > 0.5).float()  # Бинаризация масок
                    im = pred_masks.squeeze().cpu().numpy()
                    st.write('Предсказанная маска:')
                    st.image(im)

link = st.sidebar.text_input(label='Вставьте сюда ссылку на снимок')
if link is not '':
    image = Image.open(urllib.request.urlopen(link)).convert("RGB")
    st.write('Ваше изображение:')
    st.image(image)
    if model is not None:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        if model is not None:
            with torch.no_grad():
                outputs = model(image_tensor)
                pred_masks = (outputs > 0.5).float()  # Бинаризация масок
                im = pred_masks.squeeze().cpu().numpy()
                st.write('Предсказанная маска:')
                st.image(im)

