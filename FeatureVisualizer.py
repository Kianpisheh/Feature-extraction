import numpy as np
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from FeatureExtractor import FeatureExtractor
from AudioFeatureExtractor import AudioFeatureExtractor
from DataManager import DataManager
import pandas as pd
import librosa.display


plt.style.use('seaborn')


class FeatureVisualizer:

    feature_set = {'acc': 'linear Acceleration', 'mag': 'mag-akm09911', 'aud': 'audio'}
    
    def __init__(self, winsize, overlap, fs_dict, audio_sr):
        self._winsize = winsize
        self._overlap = overlap
        self._fs = fs_dict
        self._audio_sr = audio_sr
        self._data = {}
        self.t_start = None

    def set_data(self, sensor_features, audio_features):
        """Set the extracted features

           Args:
                sensor_features: a dictionary of dataframes
                audio_features: a dataframe
        """
        self._data['sensors'] = sensor_features
        self._data['audio'] = audio_features

        for i, data in enumerate(sensor_features.values()):
            self.t_start = min(self.t_start, data.iloc[0,0]) if i>0 else data.iloc[0,0]

    def show(self, features):
        """Show the given sensor data

           Args:
                features: a list of feature indicators (strings)
        """

        queries = self._query_parser(features)
        num_row = len(queries)
        num_col = 1
        ref_timestamp = 0
        for i, query in enumerate(queries):

            if query.sensor is 'audio':
                desired_sensor = self._data['audio']
                desired_feature = query.feature
                timestamp = ref_timestamp
                if query.feature != 'chroma':
                    y = desired_sensor[desired_feature + '_0']
                else:
                    y = self._get_chroma(desired_sensor)
                    
            else:
                features = self._data['sensors']
                # retrieve sensor
                desired_sensor = features[query.sensor]
                if query.sensor == 'linear Acceleration':
                    ref_timestamp = desired_sensor['timestamp']
                # retrieve feature type
                desired_feature = query.feature
                timestamp = desired_sensor['timestamp']
                # pick the data to be drawn
                if query.modifier is 'mag':
                    y = self._calc_magnitude(desired_sensor, desired_feature)
                else:
                    y = desired_sensor[desired_feature + '_0']
                
            if i == 0:
                ax = plt.subplot(num_row, num_col, i+1)
            else:
                plt.subplot(num_row, num_col, i+1, sharex=ax)

            if query.feature != 'chroma':
                plt.plot(timestamp - self.t_start, y)
            else:
                self._spectogram(np.concatenate((y, y, y, y, y), axis=1))
            plt.title(query.sensor +'_' + desired_feature)
            plt.tight_layout()

        plt.figure()
        self._spectogram(y)   
        plt.show()

    def _query_parser(self, features):
        queries = []
        for f in features:
            q = f.split('_')
            if q[0] in self.feature_set:
                query = self._Query()
                query.sensor = self.feature_set[q[0]]
                query.feature = q[1]
                if  query.sensor is not 'audio':
                    query.feature += '_' + q[2]
                query.modifier = ''
                if len(q) == 4:
                    query.modifier = q[3]
                queries.append(query)

        return queries

    @staticmethod
    def _calc_magnitude(data, feature_name):
        mag = 0
        for f in data.columns:
            if f.split('_')[0] == feature_name:
                mag = data[f].values ** 2
        
        return pd.Series(np.sqrt(mag))

    @staticmethod
    def _get_chroma(audio_features):
        chroma = np.array([])
        for i, feature in enumerate(audio_features.columns):
            if feature.split('_')[0] == 'chroma':
                print(feature)
                chroma = audio_features[feature].values.reshape(1,-1) if i==0 else np.append(chroma, audio_features[feature].values.reshape(1,-1), axis=0)

        return chroma

    @staticmethod
    def _spectogram(x):
        """Plot the given spectral signal

            Args:
                x: the input spectral signal (ndarray) shape: n_channel x sample 
        """

        plt.imshow(x)


    class _Query:
        
        def __init__(self):
            self.sensor = None
            self.feature = None
            self.modifier = None


if __name__ == '__main__':

    # load data
    audio_filename = "../Huawei/18-03-2019-17-01-19/1552928479438_audio_01.wav"
    file_path = "../Huawei/18-03-2019-17-01-19"
    data_manager = DataManager()
    data_dict, fs_dict = data_manager.read_files(file_path)
    audio_feature_extractor = AudioFeatureExtractor()
    audio = audio_feature_extractor.read_audio(audio_filename)

    # setup the feature extractors
    win_size = 0.2 # second
    overlap = 0.5  # percentage
    seq_len = 10   # number of samples per sequence
    feature_extractor = FeatureExtractor()
    feature_extractor.set_sampler(win_size, overlap, fs_dict)
    audio_feature_extractor.set_sampler(win_size, overlap)

    features = feature_extractor.extract_features(data_dict, fs_dict, type="all", n_samples=20, rt_type="flat")
    audio_features = audio_feature_extractor.extract_features(audio, n_samples=20)
    visualizer = FeatureVisualizer(win_size, overlap, fs_dict, audio_feature_extractor.sr)
    a = np.arange(12).reshape(3,4)
    
    visualizer.set_data(features, audio_features)
    visualizer.show(['acc_stat_mean_mag', 'mag_freq_var_mag', 'aud_flatness', 'aud_chroma'])
    x = 1
