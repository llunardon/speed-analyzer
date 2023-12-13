import os
import sys
import numpy as np
import argparse
import subprocess
import librosa
import cv2 as cv

import utils
import vision


def divide_spec(spec, scale, out_path):
    if not out_path.endswith("/"):
        out_path = out_path + "/"
    c_path = out_path + "c/"
    w_path = out_path + "w/"

    if not (os.path.isdir(c_path) or os.path.isdir(w_path)):
        utils.create_folder(c_path)
        utils.create_folder(w_path)

    # create spectrogram's filenames
    out_name = spec.split("/")[-1]

    # this is probably useless as a v_size of 20 likely works for every scale, will remove
    v_size = {
        'log': 10,
        'mel': 20,
        'lin': 20,
    }

    # read image and highlight split
    img = cv.imread(spec)
    thresholded = vision.highlight_split(
        img=img, low_thresh=15, high_thresh=255, h_size=3, v_size=v_size[scale])

    # --- for manual revision only
    # comparison_img = np.concatenate([img, thresholded])
    # comparison_img = cv.resize(comparison_img, (1000, 128), cv.INTER_LINEAR)
    # cv.imshow("img", comparison_img)
    # cv.waitKey(0)
    # ----------------------------

    img_splits = vision.find_splits(thresholded)
    if len(img_splits) != 0:
        vision.divide_half(img=img, filename=out_name, middle=img_splits[len(
            img_splits)//2], left_path=c_path, right_path=w_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide some spectrogram(s) in two parts, the one before the speed transition and the one after")

    # required argument: spectrogram(s)
    parser.add_argument('-i', '--input', type=str,
                        help="""Path to the spectrogram(s) to divide. It can be either
                        a folder or a single file""")

    # required argument: output folder
    parser.add_argument('-o', '--output', type=str,
                        help="""Path in which to store the output. If it doesn't exist, it
                        will be created""")

    # TODO: scale argument is probably useless, will remove
    # required argument: scale
    parser.add_argument('-s', '--scale', type=str,
                        help="""What scale is used on the y-axis of the spectrogram(s).
                        Possible options are 'log', 'mel' or 'lin'""")

    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    scale = args.scale

    # check validity of scale parameter
    if scale not in ["log", "mel", "lin"]:
        print(
            f"{scale} is not a valid scale option, see divide.py -h for help. Exiting program")
        sys.exit(1)

    if in_path.endswith(".png"):
        divide_spec(spec=in_path, scale=scale, out_path=out_path)

    elif os.path.isdir(in_path):
        spec_list = utils.collect_png_files(in_path)

        if len(spec_list) == 0:
            print(f"{in_path} doesn't contain any .png images. Exiting program.")
            sys.exit(1)

        for spec in spec_list:
            divide_spec(spec=spec, scale=scale, out_path=out_path)

    else:
        print(f"{in_path} is neither a .png file, nor a folder.")
        sys.exit(1)
