import os
import sys
import numpy as np
import argparse
import subprocess
import librosa
import cv2 as cv

import utils
import vision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide some spectrogram(s) in two parts, the one before the speed transition and the one after")

    # required argument: 'correct' spectrogram(s)
    parser.add_argument('-c', '--correct', type=str,
                        help="folder containing the spectrogram(s) labeled 'w' to divide.")

    # required argument: 'wrong' spectrogram(s)
    parser.add_argument('-w', '--wrong', type=str,
                        help="folder containing the spectrogram(s) labeled 'w' to divide.")

    # required argument: output folder
    parser.add_argument('-o', '--output', type=str,
                        help="""Path in which to store the segmented spectrogram(s). If it doesn't exist it will
                        be created. Name must be different from input folder's name""")

    # parse input arguments
    args = parser.parse_args()
    c_path = args.correct
    w_path = args.wrong
    out_path = args.output

    # correct the path adding the trailing "/" if missing
    if not out_path.endswith("/"):
        out_path = out_path + "/"
    if not c_path.endswith("/"):
        c_path = c_path + "/"
    if not w_path.endswith("/"):
        w_path = out_path + "/"

    # check validity of input parameters
    if not (os.path.isdir(c_path)) or not (os.path.isdir(w_path)):
        print(f"The input directories don't exist. Exiting program.")
        sys.exit(1)

    # create output paths for the segments
    seg_c_path = out_path + "c/"
    seg_w_path = out_path + "w/"

    if (c_path == seg_c_path or w_path == seg_w_path):
        print(f"Input and output folder must be different. Exiting program.")
        sys.exit(1)

    if not (os.path.isdir(seg_c_path) or os.path.isdir(seg_w_path)):
        utils.create_folder(seg_c_path)
        utils.create_folder(seg_w_path)

    vision.compute_segments([c_path, w_path], [seg_c_path, seg_w_path],
                            step=64, window_width=256, multiple=True, offset=0)
