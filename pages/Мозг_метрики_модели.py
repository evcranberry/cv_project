import streamlit as st

st.logo('./images/stonks.jpg', icon_image='./images/up.png', size='large')

st.title('Метрики моделей детекции опухолей головного мозга')

def display_model_info(model_type):
    st.write(f'### Процесс обучения модели: {model_type}')
    st.image(f'images/{model_type}/results.png')
    if model_type == 'последовательная модель':
        st.write('''
#### Состав датасета: 
* 2 класса результата: positive, negative  
* тренировочный датасет: 894 снимка
* тестовый датасет: 224 снимка''')
        st.write('''#### Время обучения модели: 
* суммарно 53 минуты
#### Число эпох обучения:
* суммарно 300 эпох
''')
    elif model_type == 'все вместе модель':
        st.write('''
#### Состав датасета: 
* 2 класса результата: positive, negative  
* тренировочный датасет: 894 снимка
* тестовый датасет: 224 снимка''')
        st.write('''#### Время обучения модели: 
* 50 минут
#### Число эпох обучения:
* 150
''')
    elif model_type == 'фронтальная модель':
        st.write('''
#### Состав датасета: 
* 2 класса результата: positive, negative  
* тренировочный датасет: 320 снимков
* тестовый датасет: 79 снимков''')
        st.write('''#### Время обучения модели: 
* 15 минут
#### Число эпох обучения:
* 100
''')
    elif model_type == 'саггитальная модель':
        st.write('''
#### Состав датасета: 
* 2 класса результата: positive, negative  
* тренировочный датасет: 265 снимков
* тестовый датасет: 71 снимок''')
        st.write('''#### Время обучения модели: 
* 11 минут
#### Число эпох обучения:
* 100
''')
    elif model_type == 'горизонтальная модель':
        st.write('''
#### Состав датасета: 
* 2 класса результата: positive, negative  
* тренировочный датасет: 311 снимков
* тестовый датасет: 76 снимков''')
        st.write('''#### Время обучения модели: 
* 50 минут
#### Число эпох обучения:
* 150
''')

    # Выбор метрики
    metric_options = ['F1-метрика', 'Confusion matrix', 'Precision', 'PR', 'Recall']
    selected_metric = st.selectbox("Выберите метрику:", metric_options)

    if selected_metric == 'F1-метрика':
        st.write('### График F1-метрики:')
        st.image(f'images/{model_type}/F1_curve.png')
    elif selected_metric == 'Confusion matrix':
        st.write('### Confusion matrix:')
        st.image(f'images/{model_type}/confusion_matrix.png')
    elif selected_metric == 'Precision':
        st.write('### График precision:')
        st.image(f'images/{model_type}/P_curve.png')
    elif selected_metric == 'PR':
        st.write('### График PR:')
        st.image(f'images/{model_type}/PR_curve.png')
    elif selected_metric == 'Recall':
        st.write('### График recall:')
        st.image(f'images/{model_type}/R_curve.png')

if 'model_selected' not in st.session_state:
    st.session_state.model_selected = None

# Кнопка для обучения на разных проекциях
if st.button('Обученная последовательно на разных проекциях'):
    st.session_state.model_selected = 'последовательная модель'

# Кнопка для обучения на всех проекциях сразу
if st.button('Обученная на всех проекциях сразу'):
    st.session_state.model_selected = 'все вместе модель'

# Кнопка для обучения на фронтальных проекциях
if st.button('Обученная на фронтальной проекции'):
    st.session_state.model_selected = 'фронтальная модель'

# Кнопка для обучения на горизонтальных
if st.button('Обученная на горизонтальных проекциях'):
    st.session_state.model_selected = 'горизонтальная модель'

# Кнопка для обучения на саггитальных проекциях
if st.button('Обученная на саггитальных проекциях'):
    st.session_state.model_selected = 'саггитальная модель'

# Отображение информации о модели, если она была выбрана
if st.session_state.model_selected:
    display_model_info(st.session_state.model_selected)