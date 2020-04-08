import cv2
import time
import glob
import warnings
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import functions as ft
from tensorflow import keras
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from scipy.ndimage.filters import gaussian_filter1d
from IPython.display import clear_output

model_list = ft.load_models()
videos = ["../video/actor1_neutral.mp4"]
ft.record_emotion("test", video_name = videos[0], model_list = model_list,  parse_sec = 6, mode = 1,
                   plt_pars = (4, 2, 300), TL = 0.45, name_face = False)