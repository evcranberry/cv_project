import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import time
import urllib.request
from ultralytics import YOLO


st.page_link('pages/Мозг_метрики_модели.py', label='Узнать детали обучения модели')
st.title('__Определите опухоль по снимку МРТ!🧠__')

st.logo('./images/mri.jpg', icon_image='./images/brain.jpg', size='large')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


model1 = YOLO('models/последовательная.pt')
model1.to(DEVICE)

model2 = YOLO('models/all_dataset.pt')

def get_prediction(img, model) -> int:
    start = time.time()
    results = model.predict(img) # Получаем вывод модели
    end = time.time()
    pred_images = [result.plot() for result in results]
    return end-start, pred_images

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.write('##### <- Загрузите снимок МРТ в саггитальной, горизонтальной или фронтальной проекциях, как на примере:')
ex_image = Image.open('images/example.jpg')
st.image(ex_image)

uploaded_file = st.sidebar.file_uploader(label='Загружать снимок сюда:', type=['jpeg', 'png'], accept_multiple_files=True)

st.write('Выберите модель')

if st.button('Обученная последовательно на разных проекциях'):
    model = model1
if st.button('Обученная на всех проекциях сразу'):
    model = model2

if uploaded_file is not None:
    for file in uploaded_file:
        image = Image.open(file)
        sec, results = get_prediction(image, model)
        st.write(f'''Время выполнения предсказания: __{sec:.4f} секунды__ 
    \nРезультат детекции:''')
        st.image(results, use_container_width=True)
        

link = st.sidebar.text_input(label='Вставьте сюда ссылку на снимок')
if link is not '':
    image = Image.open(urllib.request.urlopen(link)).convert("RGB")
    sec, results = get_prediction(image)
    st.session_state.predictions.append((results, image))
    for pred, img in st.session_state.predictions:
        st.write(f'''Время выполнения предсказания: __{sec:.4f} секунды__ 
\nРезультат детекции:''')
        st.image(results, use_container_width=True)

