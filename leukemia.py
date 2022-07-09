# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:23:11 2022

@author: Rushi
"""

#api libraries
from flask import Flask, request, render_template

#predicting libraries
from tensorflow.keras import optimizers, preprocessing
import tensorflow_addons as tfa
import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import os


app = Flask(__name__)

def get_model():
    global model
    DATA_PATH = "D:/RK/Marwadi University/Sem-8/Project/C-NMC_Leukemia/"
    os.chdir(DATA_PATH)

    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    LEARNING_RATE = 3e-5
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy',tfa.metrics.F1Score(num_classes=2, average='weighted')])

    
def load_image(img_path):
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(200, 200))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255
    
    return img_tensor

def predictions(img_path):
    
    new_image = load_image(img_path)
    
    preds = model.predict(new_image)
    
    prediction = np.argmax(preds)
    pct = "{:.2f}".format(np.max(preds)*100)
    if prediction == 1:
        return ["The Prediction of the sample is: ALL", pct]
    else:
        return ["The Prediction of the sample is: HEM", pct]

@app.route("/", methods=['GET','POST'])

def home():
    
    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])

def predict():
    
    if request.method == 'POST':
        
        get_model()
        
        file = request.files['file']
        
        filename = file.filename
        
        file_path = os.path.join('static', filename)
        
        file.save(file_path)
        
        print(file_path)
        product = predictions(file_path)
        str1 = str(product[1])
        value = ("Prediction Confidence Percentage is: " + str1 + "%")
        print(value)
        
        return render_template('predict.html', user_image = file_path, product = product[0], value = value) 

if __name__ == "__main__":
    app.run()
    