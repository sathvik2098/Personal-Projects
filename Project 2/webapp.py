import streamlit as st
import tensorflow as tf
import cv2# opencv-python
import numpy as np
from PIL import Image#Pillow

model = tf.keras.models.load_model('model.h5')  
def import_and_predict(image, model):
    labels = {0:'Myocardial Infarction', 1:'History of MI', 2:'Abnormal HearBeat', 3:'Normal'}
    test_img = cv2.imread(image)
    test_img = cv2.resize(test_img, (100, 100))
    tensor = tf.expand_dims(test_img, axis=0)
    pred = labels[np.argmax(model.predict(tensor))]
    return pred
def load_image(image_file):
    img = Image.open(image_file)
    img.save('1.png')
    return img

st.title("Heart Disease detection using Deep Learning")
st.write("Enter your image here")
image = st.file_uploader("Upload Image here", type=['jpg'])
submit = st.button("Predict")

if submit:
    load_image(image)
    res = import_and_predict('1.png', model)
    st.write(res)
    