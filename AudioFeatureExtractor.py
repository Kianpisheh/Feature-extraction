import librosa
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import numpy as np
import pandas as pd


class AudioFeatureExtractor:
    def __init__(self):
        self._winsize = 0
        self._overlap = 0.5
        self.audio = None
        self.sr = None
        self.duration = 0

    def read_audio(self, filepath):
        audio, self.sr = librosa.load(filepath, offset=0, sr=None)
        self.duration = (audio.shape[0] / self.sr) * 1000
        return audio

    def set_sampler(self, win_size, overlap):
        self._winsize = win_size
        self._overlap = overlap

    def _sample(self, x, i):
        start = int(self.sr * self._winsize * (i * (1 - self._overlap)))
        end = start + int(self.sr * self._winsize)
        timestamp = int((start / self.sr) * 1000)
        return x[start:end], timestamp

    def extract_features(self, audio, n_samples):

        feature_types = [
            "zero_cross",
            "mfccs_mean",
            "mfccs_std",
            "roll_off",
            "flatness",
        ]

        features = pd.DataFrame({}, columns=feature_types)
        features_fft = pd.DataFrame({}, columns=feature_types)

        FFT = np.array([])
        ds_factor = 10
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"audio_feature:{i}")

            # take sample
            sample, timestamp = self._sample(audio, i)
            feature = {}
            feature["timestamp"] = timestamp
            fft_ = self._calc_fft(sample, ds_factor)
            fft_feat = np.append(np.array([timestamp]), fft_)
            FFT = np.append(FFT, fft_feat.reshape(1, -1)).reshape(-1, fft_feat.shape[0])
            # for f in range(fft_.shape[0]):
            #     feature_fft[f'fft_{f}'] = fft_[f]
            feature["zero_cross"] = np.mean(librosa.zero_crossings(sample, pad=False))
            mfccs = librosa.feature.mfcc(sample, sr=self.sr)
            # features["mfcc"] = sklearn.preprocessing.scale(mfccs, axis=1)
            feature["mfccs_mean"] = mfccs.mean()
            feature["mfccs_std"] = mfccs.std()
            feature["roll_off"] = librosa.feature.spectral_rolloff(
                sample, sr=self.sr, roll_percent=0.2
            ).mean()
            feature["flatness"] = librosa.feature.spectral_flatness(sample).mean()
            chroma = librosa.feature.chroma_stft(sample, sr=self.sr).mean(axis=1)
            for c in range(chroma.shape[0]):
                feature["chroma_" + str(c)] = chroma[c]
            mfcc_means = mfccs.mean(axis=1)
            for m in range(mfcc_means.shape[0]):
                feature["mfcc_" + str(m)] = mfcc_means[m]

            features = features.append(pd.DataFrame(feature, index=[0]))

        fft_columns = []
        fft_columns.append("timestamp")
        for i in range(FFT.shape[1] - 1):
            fft_columns.append(f"fft_{i}")
        features_fft = pd.DataFrame(FFT, columns=fft_columns)
        return features, features_fft

    def _calc_fft(self, x, ds_factor):
        N = x.shape[0]
        win = np.hanning(N + 1)[:-1]
        windowed_sample = win * x
        fft_ = np.fft.fft(windowed_sample)[: N // 2 + 1 : -1]
        return np.abs(fft_)

    def show_chromogram(self, chroma):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis="chroma", x_axis="time")
        plt.colorbar()
        plt.title("Chromagram")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # load data
    audio_filename = "../Huawei/18-03-2019-17-01-19/1552928479438_audio_01.wav"
    audio_feature_extractor = AudioFeatureExtractor()
    audio = audio_feature_extractor.read_audio(audio_filename)

    # setup the feature extractors
    win_size = 0.2  # second
    overlap = 0.5  # percentage
    audio_feature_extractor.set_sampler(win_size, overlap)
    audio_features = audio_feature_extractor.extract_features(audio, n_samples=15)
