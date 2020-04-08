from tensorflow import keras
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras import backend as K
K.clear_session()

from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from flask import render_template, Flask, flash, request, redirect, url_for, request
# from werkzeug import secure_filename
from flaskexample import app

import functions as ft
import pandas as pd
import psycopg2
import os
import glob
import cv2
import warnings
import face_recognition
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def produce_videos(filePath, model_list, parse_sec = 3):
    video = cv2.VideoCapture(filePath)
    face_locations = []
    # reduce the resolution to 0.1
    fraction = 0.1 
    

    process = 0
    pre_Face = None
    left = -1
    emt_smth_period = 0.25
    
#     plt_width, plt_height,plt_dpi = plt_pars
    fig = plt.figure(figsize = (1.28, 0.72), dpi = 1000)
    plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.23)
    ax = fig.add_subplot(111)
    x, y = [0], [0]
    img = ft.process_figure(fig, x, y)
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = video.get(cv2.CAP_PROP_FPS)
    process_period = int(fps/parse_sec)
    
    
    face_file = filePath[:-4] + '_parse_face.mp4'
    emotion_file = filePath[:-4] +'_parse_emotion.mp4'
    out = cv2.VideoWriter(face_file, cv2.VideoWriter_fourcc(*'avc1'), int(fps), (frame_width, frame_height))
    out_plt = cv2.VideoWriter(emotion_file, cv2.VideoWriter_fourcc(*'avc1'), int(fps)
                              , (frame_width, frame_height))
    a = filePath[:-4] + '_parse_face.mp4'
    print(a)
    while True:
        ret, frame = video.read()
        fps = video.get(cv2.CAP_PROP_FPS)
        time = process/fps
        if not ret: break
            
        if process % process_period == 0:
            small_frame = cv2.resize(frame, (0, 0), fx = fraction, fy = fraction)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if(len(face_locations) != 0):
                main_face = ft.get_main_face(face_locations, pre_Face)
                pre_Face = main_face
                face_encoding = face_recognition.face_encodings(rgb_small_frame, [main_face])
                emo_string, score = ft.judge(rgb_small_frame, main_face, model_list)
                (top, right, bottom, left) = main_face
                top *= int(1/fraction)
                right *= int(1/fraction)
                bottom *= int(1/fraction)
                left *= int(1/fraction)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                x.append(time)
                y.append(score) 
                smth_y = gaussian_filter1d(np.array(y), sigma = emt_smth_period* fps*0.2)
                img = ft.process_figure(fig, x, smth_y)
                

        if left != -1:
            cv2.rectangle(frame, (left, bottom - 40), (right, bottom + 40), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "%s user" %(emo_string), (left + 6, bottom - 10), font, 1.1, (255, 255, 255), 2)
            cv2.putText(frame, "score = %.2f" %(score), (left + 6, bottom + 30), font, 1.1, (255, 255, 255), 2)
    
        out_plt.write(img)
        out.write(frame)
        process += 1    
    plt.close()    
    video.release()
    out.release()
    out_plt.release()
    return face_file, emotion_file

# app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/')
def home():
    return render_template("HomePage.html")

@app.route('/Example1')
def example1():
    return render_template("Example1.html")

@app.route('/Example2')
def example2():
    return render_template("Example2.html")


ALLOWED_EXTENSIONS = set(['mp4', 'mov', 'avi'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# handle upload
UPLOAD_FOLDER = "./flaskexample/static/user_upload/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/user', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return ('', 204)

        file = request.files['file']
        # do nothing is the file is not valid
        if (file.filename == '' or not allowed_file(file.filename)
         or request.content_length >= 8 * 1024 * 1024):
            return ('', 204)

        # filename = secure_filename(file.filename)
        filePath = app.config['UPLOAD_FOLDER'] + file.filename
        file.save(filePath)
        model_path = "./flaskexample/static/Trained Model/"
        model_list = ft.load_models(model_path)
        face_file, emotion_file = produce_videos(filePath, model_list, parse_sec = 3)
        face_file = ".." + face_file[14:]
        emotion_file = ".." + emotion_file[14:]
    
    # return type(face_file)
    return render_template("user.html", face_file = face_file, emotion_file = emotion_file)






