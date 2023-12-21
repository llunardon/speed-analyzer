import os
import sys
import numpy as np
import argparse
import subprocess
import librosa

import utils
import vision


def wav2spec(sample, scale, out_path, width_res=256):
    if not out_path.endswith("/"):
        out_path = out_path + "/"
    if not os.path.isdir(out_path):
        utils.create_folder(out_path)

    # create spectrogram's filename
    out_name = sample.split("/")
    out_name = out_path + out_name[-1].split(".")[0] + '.png'

    duration = round(librosa.get_duration(filename=sample), 2)
    if duration < 1:
        print(f"file {sample} is too short. Skipping to next one..")
        return

    # mel-spec
    if scale == "mel":
        y, sr = librosa.load(sample, sr=None)

        vision.mel_spectrogram_image(
            y, sr, out_name, hop_length=512, n_mels=128, dimensions=(int(duration*width_res), 128), save=True)

    else:
        width = str(duration*width_res)
        height = str(128)

        # lin-spec
        if scale == "lin":
            subprocess.call([
                'sox',
                sample,
                '-n', 'spectrogram',
                '-y', height,
                '-x', width,
                '-r',
                '-m',
                '-R', '0:20k',
                '-o', out_name,
            ])

        # log-spec
        elif scale == "log":
            subprocess.call([
                'sox',
                sample,
                '-n', 'spectrogram',
                '-y', height,
                '-x', width,
                '-r',
                '-m',
                '-L',
                '-R', '0:20k',
                '-o', out_name,
            ])

    return out_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .wav file(s) to .png spectrograms")

    # required argument: audio sample
    parser.add_argument('-i', '--input', type=str,
                        help="""Path to the audio sample(s) to analyze. It can be either
                        a folder or a single file""")

    # required argument: output folder
    parser.add_argument('-o', '--output', type=str,
                        help="""Path in which to store the output. If it doesn't exist, it
                        will be created""")

    # required argument: scale
    parser.add_argument('-s', '--scale', type=str,
                        help="""What scale to use on the y-axis of the spectrogram.
                        possible options are 'log', 'mel' or 'lin'""")

    # optional arguments: resolution of x-axis (pixels per second)
    parser.add_argument('-w', '--width', nargs='?', type=int, default=256,
                        help="What resolution to use on the x-axis, in pixels/s. Default is 256.")

    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    scale = args.scale
    width_res = args.width

    # check validity of scale parameter
    if scale not in ["log", "mel", "lin"]:
        print(
            f"{scale} is not a valid scale option, see wav2spec.py -h for help. Exiting program")
        sys.exit(1)

    if in_path.endswith(".wav"):
        wav2spec(sample=in_path, scale=scale, out_path=out_path, width_res=width_res)

    elif os.path.isdir(in_path):
        sample_list = utils.collect_audio_files(in_path)

        if len(sample_list) == 0:
            print(f"{in_path} doesn't contain any .wav samples. Exiting program.")
            sys.exit(1)

        for sample in sample_list:
            wav2spec(sample=sample, scale=scale, out_path=out_path, width_res=width_res)

    else:
        print(f"{in_path} is neither a .wav file, nor a folder.")
        sys.exit(1)
