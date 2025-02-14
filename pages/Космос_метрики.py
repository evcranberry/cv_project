import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Метрики моделей сегментации')
st.write('Выберите модель:')

df = pd.read_csv('images/seg_data/result_model_Unet_resnet101.csv', index_col=0)
if st.button('Unet'):
    df_loss = df.drop(columns=['train_iou', 'valid_iou'])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_loss)
    plt.title(f"Loss")
        
    st.pyplot(plt)

    df_iou = df.drop(columns=['train_loss', 'valid_loss'])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_iou)
    plt.title(f"IoU")
        
    st.pyplot(plt)

df_seg = pd.read_csv('images/seg_data/result_model_segformer_2.csv', index_col=0)

if st.button('Segformer'):
    df_loss_seg = df_seg.drop(columns=['train_iou', 'valid_iou'])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_loss_seg)
    plt.title(f"Loss")
        
    st.pyplot(plt)

    df_iou_seg = df_seg.drop(columns=['train_loss', 'valid_loss'])
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_iou_seg)
    plt.title(f"IoU")
        
    st.pyplot(plt)

