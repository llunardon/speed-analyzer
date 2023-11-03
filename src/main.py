import os
import sys
import numpy as np

import utils
import vision

from scipy.io import wavfile
from keras.models import load_model
import librosa

# load the keras classifiers
binary_model = load_model('../models-def/model-binary-separatechannels/')
four_classes_model = load_model(
    '../models-def/model-4classes-separatechannels/')

# spectrogram settings
hop_length = 256  # number of samples per time-step in spectrogram
n_mels = 128  # number of bins in spectrogram. Height of image
time_steps = 512  # number of time-steps.

# settings for the scanning of the spectrogram
step = 32
offset = 0
window_width = 256

# speedup corresponding to the label output by the second classifier
speedup_dict = {
    0: "double",
    1: "half",
    2: "quadruple",
    3: "quarter"
}

utils.create_folder("../testing/")

# read the audio file
sample = sys.argv[1]
if not (sample.endswith(".wav")):
    print("The input file is not a .wav file")

fs, data = wavfile.read(sample)

# for each channel:
for i in range(0, data.shape[1]):
    log_filename = "../testing/output-ch" + str(i) + ".txt"
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Filename: {sample}\n")
        print(f"Created file {log_filename}")

    # create folders where to save the separate audio and segmented spectrograms
    channel_path = '../testing/ch' + str(i) + "/"
    segments_path = '../testing/segments_ch' + str(i) + "/"
    utils.create_folder(channel_path)
    utils.create_folder(segments_path)

    # save audio of channel
    channel_filename = channel_path + "ch" + str(i) + ".wav"
    wavfile.write(channel_filename, fs, data[:, i])

    # load audio of channel and get the duration
    y, sr = librosa.load(channel_filename, offset=0.0, duration=None)
    duration = round(librosa.get_duration(filename=sample), 2)

    with open(log_filename, 'a') as log_file:
        log_file.write(f"Duration: {duration}s\n")

    # save spectrogram
    spec_name = channel_path + "spec_ch" + str(i) + '.png'
    spectrum = vision.spectrogram_image(
        y, sr=sr, out=spec_name, hop_length=hop_length, n_mels=n_mels, save=True)
    height, width = spectrum.shape[:2]

    if width < 256:
        print("The audio file is too short to be analyzed")
        break

    # scan the whole spectrogram and divide it in segments (windows)
    for i in range(0, width//step):
        if offset + i*step + window_width < width:
            window = spectrum[0:height, offset +
                              i*step:offset+(i*step)+window_width]
            window = np.expand_dims(window, axis=0)

            # classify the window
            binary_label = np.argmax(binary_model.predict(window, verbose=0))

            if binary_label == 1:  # irregularity detected
                timestamp = round(duration * (i * step) / width, 2)

                # classify the speedup factor
                speedup_label = np.argmax(
                    four_classes_model.predict(window, verbose=0))
                print(
                    f"Segment {i}, time: {str(timestamp)}s, speedup: {speedup_dict[speedup_label]}")
                with open(log_filename, 'a') as log_file:
                    log_file.write(
                        f"Segment {i}, time: {str(timestamp)}s, speedup: {speedup_dict[speedup_label]}\n")

    # save the windows for manual analysis
    vision.compute_segments([channel_path], [segments_path], step=step,
                            window_width=window_width, overwrite=True, multiple=True, offset=0)
