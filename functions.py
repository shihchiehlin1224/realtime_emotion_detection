import cv2
import os
import time
import glob
import warnings
import face_recognition
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def load_models(model_path):
    neg_model = keras.models.load_model(model_path + "mini_Xception_negative_210_epoch.h5")
    neu_model = keras.models.load_model(model_path + "mini_Xception_neutral_100_epoch.h5")
    pos_model = keras.models.load_model(model_path + "mini_Xception_positive_60_epoch.h5")
    model_list = [neg_model, neu_model, pos_model]
    return model_list


def close_cam(ved):
    ved.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def process_input(x):
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

def feed_model(img, shape):
    oriimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(oriimg, shape)
    img = np.expand_dims(img, axis = -1)
    return img

def overlap(face_location, pre_Face):
    new_top = max(pre_Face[0], face_location[0])
    new_left = max(pre_Face[3], face_location[3])
    new_right = min(pre_Face[1], face_location[1])
    new_bot = min(pre_Face[2], face_location[2])
    area = 0
    if(new_right > new_left) and (new_bot > new_top):
        area = (new_bot - new_top)*(new_right - new_left)
    return area


def get_main_face(face_locations, pre_Face):
    if pre_Face != None:
        main_Face = sorted(face_locations, key = lambda edges: overlap(edges, pre_Face),
                   reverse = True)[0]
        if overlap(main_Face, pre_Face) == 0:
            main_Face = pre_Face 
    else:
        main_Face = sorted(face_locations, key = lambda edges: (edges[2] - edges[0])*(edges[1] - edges[3]),
                   reverse = True)[0]
    return main_Face


def model_pred(img, model_list):
    emotion = np.zeros(3)
    # The shape of img is 1 x 48 x 48 x 1
    for i, model in enumerate(model_list):
        emotion[i] = model.predict(img)[0][1]
#     print(np.argmax(emo_string))
    emo_string = "Negative"
    if np.argmax(emotion) == 1:
        emo_string = "Neutral"
    elif np.argmax(emotion) == 2:
        emo_string = "Positive"
    return emotion, emo_string
        

# model_list = [neg_model, neu_model, pos_model]
def judge(frame, edges, model_list):
    img = feed_model(frame[edges[0]: edges[2], edges[3]: edges[1]], (48, 48))
    img = process_input(img)
    face_image = np.array(img)
    # face_image shape 1 x 48 x 48 x 1
    face_image = face_image[np.newaxis]
    emotion, emo_string = model_pred(face_image, model_list)
    score_weight = np.array([-1, 0, 1])
    score = np.dot(score_weight, emotion)
    return emo_string, score

def load_members():
    image_files = glob.glob("./Members/*.png")
    known_face_names = [file[10:-4] for file in image_files]
    known_face_encodings = []

    for file in image_files:
        img = face_recognition.load_image_file(file)
        known_face_encodings.append(face_recognition.face_encodings(img)[0])
    return known_face_names, known_face_encodings



def process_figure(fig, x, y, neutral_range = 0.25):
    canvas = FigureCanvasAgg(fig)
    avg_emo = np.ones(len(x))*np.mean(y)
    ax = fig.add_subplot(111)
    ax.cla()
    
    neutral_range = 0.25
    marker = dict(marker='o', markersize = 2, markeredgewidth = 0)
    ax.plot(x, y, color = 'r', linewidth = 0.3, **marker)
    ax.plot(x, avg_emo,'--k', linewidth = 0.3)
    ax.plot([-1, -0.5], [0, 0],'r', linewidth = 2,  alpha = 0.2)
    
    ax.set_xlabel("Time(s)", fontsize = 4, labelpad = 1)
    ax.set_ylabel("Emotion", fontsize = 4, labelpad = 1)
    
    if max(x) < 0.05:
        ax.set_xlim(-0.05, 0.05)
    else:
        ax.set_xlim(-0.05, max(x))


    ax.legend(['Emotion', 'Avg_emotion', 'Neutral Zone'], fontsize = 3, frameon = False)
    ax.tick_params(which = "major", labelsize = 4, length = 1,
                    width = 1, direction = 'in', grid_linestyle = "dashed", grid_linewidth = 0.3)
    ax.axhspan(-0.25, 0.25, facecolor = 'r', alpha = 0.05)
    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5) 
    ax.tick_params(direction = 'in', length = 2, width = 0.5, colors = 'k')
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    img = cv2.cvtColor(X, cv2.COLOR_RGB2BGR)
    return img






