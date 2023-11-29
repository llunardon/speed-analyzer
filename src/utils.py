import os
import numpy as np
import shutil
import math
import random


def delete_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_folder(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder {path}")
    else:
        if overwrite == False:
            print(f"Couldn't substitute folder because overwrite is set to False")
        else:
            delete_folder(path)
            print(f"Substituted folder {path}")


def collect_audio_files(path):
    ret_list = []

    for (root, dirs, filenames) in os.walk(path):
        for f in filenames:
            if f.endswith('.wav'):
                ret_list.append(os.path.join(root, f))

    return ret_list


def collect_png_files(path):
    ret_list = []

    for (root, dirs, filenames) in os.walk(path):
        for f in filenames:
            if f.endswith('.png'):
                ret_list.append(os.path.join(root, f))

    return ret_list


"""
    inputs: in_paths  ---> list of paths, where every path corresponds to a different label
            out_paths ---> names of training, validation and test folders, in this order
              ratios  ---> what fraction of the dataset goes into training, validation and testing
              seed    ---> seed used for randomization 

              NOTE: this function is only used in the Colab Notebook for now
"""


def create_dataset(in_paths, labels, out_paths, ratios, seed):
    if len(ratios) != 3 or len(out_paths) != 3:
        print("Output configuration is wrong")
        return

    if np.sum(ratios) > 1.0:
        print("Sum of ratios must be less than 1")
        return

    for out_path in out_paths:
        create_folder(out_path)

    for label_index, in_path in enumerate(in_paths):
        label = labels[label_index]

        for (root, dirs, files) in os.walk(in_path, topdown=True):
            # number of elements in each split
            n_train = math.floor(ratios[0] * len(files))
            n_valid = math.floor(ratios[1] * len(files))
            n_test = len(files) - (n_train + n_valid)

            train_files = []
            valid_files = []
            test_files = []

            # create list of random indexes and shuffle it
            indexes = list(range(0, len(files)))
            random.Random(seed).shuffle(indexes)

            for j in range(0, len(files)):
                if j < n_train:
                    train_files.append(files[indexes[j]])
                elif n_train <= j < n_train + n_valid:
                    valid_files.append(files[indexes[j]])
                else:
                    test_files.append(files[indexes[j]])

            for i, out_path in enumerate(out_paths):
                create_folder(out_path + label + "/")

                if i == 0:
                    for filename in train_files:
                        shutil.copyfile(in_path + filename,
                                        out_path + label + "/" + filename)
                elif i == 1:
                    for filename in valid_files:
                        shutil.copyfile(in_path + filename,
                                        out_path + label + "/" + filename)
                else:
                    for filename in test_files:
                        shutil.copyfile(in_path + filename,
                                        out_path + label + "/" + filename)
