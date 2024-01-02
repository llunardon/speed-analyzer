import os
import sys
import numpy as np
import argparse
import subprocess
import librosa
from scipy.io import wavfile

import utils
import vision


def split_channels(sample, out_path):
    if not out_path.endswith("/"):
        out_path = out_path + "/"
    if not os.path.isdir(out_path):
        utils.create_folder(out_path)

    # read audio sample
    fs, data = wavfile.read(sample)

    # sample is mono
    if len(data.shape) == 1:
        data.shape = (data.shape[0], 1)

    for i in range(0, data.shape[1]):
        # save audio of channel
        channel_filename = out_path + \
            sample.split("/")[-1][0:-3] + "ch" + str(i) + ".wav"
        wavfile.write(channel_filename, fs, data[:, i])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .wav file(s) to .png spectrograms")

    # required argument: audio sample
    parser.add_argument('-i', '--input', type=str,
                        help="""Path to the audio sample(s) to divide in channels. It can be either
                        a folder or a single file""")

    # required argument: output folder
    parser.add_argument('-o', '--output', type=str,
                        help="""Path in which to store the output. If it doesn't exist, it
                        will be created""")

    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    if in_path.endswith(".wav"):
        split_channels(sample=in_path, out_path=out_path)

    elif os.path.isdir(in_path):
        sample_list = utils.collect_audio_files(in_path)

        if len(sample_list) == 0:
            print(f"{in_path} doesn't contain any .wav samples. Exiting program.")
            sys.exit(1)

        for sample in sample_list:
            split_channels(sample=sample, out_path=out_path)

    else:
        print(f"{in_path} is neither a .wav file, nor a folder.")
        sys.exit(1)
