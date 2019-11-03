import numpy as np
import pandas as pd
import os
import librosa

# import audioread
import scipy.io.wavfile
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import librosa.display
import pickle

from DataManager import DataManager
from AudioFeatureExtractor import AudioFeatureExtractor
from FeatureExtractor import FeatureExtractor

# from MHMM import MHMM
# from generator import generator

load_features = False

# load data
audio_filename = "./data/audio.wav"
file_path = "./data"
data_manager = DataManager()
data_dict, fs_dict = data_manager.read_files(file_path)
audio_feature_extractor = AudioFeatureExtractor()
audio = audio_feature_extractor.read_audio(audio_filename)

# setup the feature extractors
T = 300
TOLERENCE = 0.5  # second
WIN_SIZE = 0.2  # second
OVERLAP = 0.5  # percentage
NUM_SAMPLES = np.int((T - OVERLAP * WIN_SIZE) // (WIN_SIZE * (1 - OVERLAP)))


feature_extractor = FeatureExtractor()
feature_extractor.set_sampler(WIN_SIZE, OVERLAP, TOLERENCE, fs_dict)
audio_feature_extractor.set_sampler(WIN_SIZE, OVERLAP)

# extract features
feature = feature_extractor.extract_features(
    data_dict, fs_dict, type="all", n_samples=NUM_SAMPLES, rt_type="flat"
)
FeatureExtractor.save(feature)
audio_features, fft_ = audio_feature_extractor.extract_features(
    audio, n_samples=NUM_SAMPLES
)
FeatureExtractor.save({"audio": audio_features})
FeatureExtractor.save({"fft": fft_})
x = 1
