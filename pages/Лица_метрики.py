import streamlit as st

st.title('Метрики модели детекции лиц')
metric_options = ['F1-метрика', 'Confusion matrix', 'Confusion matrix нормализованная', 'Precision', 'PR', 'Recall', 'Labels', 'Labels correlogram']
selected_metric = st.selectbox("Выберите метрику:", metric_options)

if selected_metric == 'F1-метрика':
    st.write('### График F1-метрики:')
    st.image(f'sergio/models_metrics/faces_plots/F1_curve.png')
elif selected_metric == 'Confusion matrix':
    st.write('### Confusion matrix:')
    st.image(f'sergio/models_metrics/faces_plots/confusion_matrix.png')
elif selected_metric == 'Confusion matrix нормализованная':
    st.write('### Confusion matrix нормализованная:')
    st.image(f'sergio/models_metrics/faces_plots/confusion_matrix_normalized.png')
elif selected_metric == 'Precision':
    st.write('### График precision:')
    st.image(f'sergio/models_metrics/faces_plots/P_curve.png')
elif selected_metric == 'PR':
    st.write('### График PR:')
    st.image(f'sergio/models_metrics/faces_plots/PR_curve.png')
elif selected_metric == 'Recall':
    st.write('### График recall:')
    st.image(f'sergio/models_metrics/faces_plots/R_curve.png')
elif selected_metric == 'Labels':
    st.write('### Labels:')
    st.image(f'sergio/models_metrics/faces_plots/labels.jpg')
elif selected_metric == 'Labels correlogram':
    st.write('### Labels correlogram:')
    st.image(f'sergio/models_metrics/faces_plots/labels_correlogram.jpg')
