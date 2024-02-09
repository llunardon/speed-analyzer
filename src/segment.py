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
        description="Extract multiple fixed-width segments from the spectrograms, scanning them left-to-right.")

    # required argument: folder containing the spectrogram(s)
    parser.add_argument('-i', '--input', type=str,
                        help="Folder containing the spectrogram(s) to be segmented")

    # required argument: output folder
    parser.add_argument('-o', '--output', type=str,
                        help="""Path in which to store the segmented spectrogram(s). If it doesn't exist it will
                        be created. Name must be different from input folder's name""")

    # optional arguments: width of each segment
    parser.add_argument('-w', '--width', nargs='?', type=int, default=256,
                        help="How wide each segment should be. Default is 256 pixels.")

    # optional arguments: step
    parser.add_argument('-j', '--jump', nargs='?', type=int, default=64,
                        help="Step to use for the scanning of the spectrum. Default is 64 pixels.")

    # parse input arguments
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output
    window_width = args.width
    step = args.jump

    # correct the path adding the trailing "/" if missing
    if not in_path.endswith("/"):
        w_path = out_path + "/"
    if not out_path.endswith("/"):
        out_path = out_path + "/"

    c_path = in_path + "c/"
    w_path = in_path + "w/"

    # check validity of input parameters
    if not (os.path.isdir(c_path)) or not (os.path.isdir(w_path)):
        print(f"""The input directories don't exist, or they are
        not named 'c' and 'w'. Exiting program.""")
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
                            step=step, window_width=window_width, multiple=True, offset=128)
