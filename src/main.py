import os
import sys
import numpy as np
import argparse

import utils
import vision

from scipy.io import wavfile
from keras.models import load_model
import librosa


def analyze_speed(sample, binary_model_path, four_classes_model_path):
    # load the keras classifiers
    binary_model = load_model(binary_model_path)
    four_classes_model = load_model(four_classes_model_path)

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

    # create output folder
    out_path = os.path.dirname(os.path.realpath('__file__')) + "/output/"
    utils.create_folder(out_path)

    # read audio sample
    fs, data = wavfile.read(sample)

    # for each channel:
    for i in range(0, data.shape[1]):
        log_filename = out_path + "output-ch" + str(i) + ".txt"
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Filename: {sample}\n")
            print(f"Created file {log_filename}")

        # create folders where to save the separate audio and segmented spectrograms
        channel_path = out_path + "ch" + str(i) + "/"
        segments_path = out_path + "segments_ch" + str(i) + "/"
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
                window = spectrum[0:height, offset+i *
                                  step:offset+(i*step)+window_width]
                window = np.expand_dims(window, axis=0)

                # classify the window
                binary_label = np.argmax(
                    binary_model.predict(window, verbose=0))

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
                                window_width=window_width, overwrite=True, multiple=True, offset=offset)


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser(
        description="Analyze a digitised magnetic audio tape and detect irregularities in the playback speed.")

    # required arguments: audio sample
    parser.add_argument('-i', '--input', type=str,
                        help='path to the audio sample to analyze')

    # optional arguments: path to models
    parser.add_argument('-b', '--bin_model', type=str, nargs='?',
                        default='models-def/model-binary-separatechannels/', help='path to the binary model directory')
    parser.add_argument('-f', '--four_model', type=str, nargs='?',
                        default='models-def/model-4classes-separatechannels/', help='path to the four-classes model directory')

    args = parser.parse_args()

    # read the input parameters
    sample = args.input
    binary_model_path = args.bin_model
    four_classes_model_path = args.four_model

    if not (sample.endswith(".wav")):
        print("The input file is not a .wav file")
    else:
        analyze_speed(sample, binary_model_path, four_classes_model_path)
