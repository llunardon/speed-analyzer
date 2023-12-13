import os
import sys
import numpy as np
import argparse

import utils
import vision
from wav2spec import wav2spec

import cv2 as cv
from scipy.io import wavfile
from keras.models import load_model
import librosa


def analyze_speed(sample, scale, binary_model_path, four_classes_model_path):
    # load the keras classifiers
    binary_model = load_model(binary_model_path)
    four_classes_model = load_model(four_classes_model_path)

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
    print(data.shape)

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

        # save spectrogram and load it into numpy array
        spec_name = wav2spec(channel_filename, scale, channel_path)
        spectrum = cv.imread(spec_name, cv.IMREAD_GRAYSCALE)
        height, width = spectrum.shape[:2]

        if width < 256:
            print("The audio file is too short to be analyzed")
            break

        # settings for the scanning of the spectrogram
        step = 32
        offset = 0
        window_width = 256

        # scan the whole spectrogram and divide it in windows (segments)
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
                                window_width=window_width, multiple=True, offset=offset)


if __name__ == "__main__":
    script_path = utils.get_script_path()
    models_path = os.path.dirname(script_path) + "/models/"

    # parse input arguments
    parser = argparse.ArgumentParser(
        description="""Analyze a digitised magnetic audio tape and detect discrepancies 
        between the recording speed and the playback speed.""")

    # required arguments: audio sample
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the audio sample to analyze.')

    # required arguments: scale for y-axis
    parser.add_argument('-s', '--scale', type=str,
                        help="""Which scale to use for the y-axis of the spectrogram. Also switches to the
                        correct model to use for the classification. Valid choices are 'log', 'lin' and 'mel.""")

    args = parser.parse_args()

    # read the input parameters and check for correctness
    sample = args.input
    scale = args.scale

    if not (sample.endswith(".wav")):
        print("The input file is not a .wav file.")
        sys.exit(1)

    if not (scale in ['lin', 'log', 'mel']):
        print("The scale chosen is not valid. See main.py -h for information.")

    # define paths to models
    binary_models = {
        'lin': models_path + "model-binary-lin/",
        'log': models_path + "model-binary-log/",
        'mel': models_path + "model-binary-mel/"
    }

    four_classes_models = {
        'lin': models_path + "model-4c-lin/",
        'log': models_path + "model-4c-log/",
        'mel': models_path + "model-4c-mel/"
    }

    analyze_speed(
        sample, scale, binary_models[scale], four_classes_models[scale])
