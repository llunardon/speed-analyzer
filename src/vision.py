import cv2 as cv
import numpy as np
import os
import utils
import librosa
import skimage.io


"""
    extract the spectrogram of an audio sample loaded with librosa.load
"""


def mel_spectrogram_image(y, sr, out, hop_length, n_mels, dimensions, save):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*4, hop_length=hop_length, fmax=20000)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = cv.resize(img, dimensions, cv.INTER_LINEAR)

    # save as PNG and return numpy array
    if save == True:
        skimage.io.imsave(out, img)

    return img


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min

    return X_scaled


"""
    takes in input a spectrogram and highlights the split regions
    returns an image of the same dimensions as the original, with the ROI highlighted in black
"""


def highlight_split(img, low_thresh=3, high_thresh=255, h_size=3, v_size=20):
    # threshold the original image and extract vertical lines using vertical morphological operator
    ret, thresh1 = cv.threshold(img, low_thresh, high_thresh, cv.THRESH_BINARY)
    verticalStructure = cv.getStructuringElement(
        cv.MORPH_RECT, (h_size, v_size))
    ret_img = cv.dilate(thresh1, verticalStructure)

    return ret_img


"""
    takes in input a thresholded image (pixels are either 0 or 255)
    returns a list of lists, with every list being an interval of x coordinates that make a black band
"""


def find_splits(img):
    height, width = img.shape[:2]
    list_splits = []
    i = 0

    for j in range(0, height-1):
        # start from the last seen black column if already encountered
        if len(list_splits) > 1:
            i = list_splits[-1][-1]

        while (i < width):
            # found a black pixel: append x coordinate at the end of the ROI
            if not (img[j][i].any()):
                start_roi = i

                while not (img[j][i]).any():
                    i += 1

                end_roi = i

                list_splits.append([start_roi, end_roi])
                break

            i += 1

    return list_splits


"""
    takes in input an image and the middle region and saves the two halfs in separate files
    middle = list of two x coordinates, which mark the start and the end of the middle region
"""


def divide_half(img, filename, middle, left_path, right_path):
    height, width = img.shape[:2]

    left_roi = img[0:height, 0:middle[0]]
    right_roi = img[0:height, middle[1]:width-1]

    left_filename = left_path + "c_" + filename
    right_filename = right_path + "w_" + filename

    cv.imwrite(left_filename, left_roi)
    cv.imwrite(right_filename, right_roi)

    return [left_roi, right_roi]


"""
    takes in input an image (numpy array) and divides it into segments
    returns a list that contains all the segments

    multiple: boolean value. if set to false, only save the first segment of the image
    offset: how many pixels from the left to use as starting point
"""


def segment(img, step, window_width, multiple, offset):
    height, width = img.shape[:2]
    ret_list = []

    if multiple == False:
        window = img[0:height, offset:offset+window_width]
        ret_list.append(window)
    else:
        for i in range(0, width//step):
            if offset + i*step + window_width < width:
                window = img[0:height, offset + i *
                             step:offset+(i*step)+window_width]
                ret_list.append(window)

    return ret_list


"""
    in_paths: list of input folders
    out_paths: list of output folders
    multiple: boolean value. if set to false, only save the first segment of the image
    offset: how many pixels from the left to use as starting point (used to skip most of the glissando region)
"""


def compute_segments(in_paths, out_paths, step, window_width, overwrite, multiple, offset):
    if len(in_paths) == 0:
        return

    if len(in_paths) != len(out_paths):
        print(
            f"Length of input list is {len(in_paths)} while length of output list is {len(out_paths)}")
        return

    for path in out_paths:
        if overwrite == True:
            utils.create_folder(path)
        else:
            print(
                f"Folder {path} already exists and overwrite is set to False")

    for i, in_path in enumerate(in_paths):
        for (root, dirs, files) in os.walk(in_path, topdown=True):
            for filename in files:
                if filename.endswith('.png'):
                    img = cv.imread(os.path.join(in_path + filename))

                    seg_list = segment(
                        img, step, window_width, multiple, offset)
                    for j, seg in enumerate(seg_list):
                        out_name = out_paths[i] + \
                            filename[0:-4] + "_" + str(j) + ".png"
                        seg = cv.cvtColor(seg, cv.COLOR_RGB2GRAY)
                        cv.imwrite(out_name, seg)
