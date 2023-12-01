import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# モデルの読み込み
model = tf.keras.models.load_model('tensor_number_model.h5')

# Streamlitアプリのタイトル
st.title('Digit Recognition App')

# ユーザーがアップロードした画像を読み込む
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('L'))  # グレースケールに変換
    img = img.reshape(1, 28, 28, 1)  # モデルの入力形状に合わせる
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    # 予測結果を表示
    st.write(f"Predicted Digit: {predicted_digit}")
    st.bar_chart(prediction.ravel())
