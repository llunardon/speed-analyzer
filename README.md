# MPAI Audio Analyzer: Speed Irregularities Detector

### Usage:

```
    python3 src/main.py -i "wav_file_path" -b "binary_model_path" -f "4classes_model_path"
```

- **Note:** The arguments `-b` and `-f` are optional, if not provided the default models used are
  `model-binary-separatechannels` and `model-4classes-separatechannels` in the folder `models-def/`

---

The script generates a folder called `output` in the project directory, that contains:

- a log file for each channel describing the irregularities found, with their relative timestamps
- a folder for each channel containing the respective .wav file and spectrogram
- a folder for each channel containing the respective segmented spectrogram

The output folders are only for manual revision, they can be eliminated in the final version of the program

---

### Dataset Extraction

- **Note:** the script `wav2spec.py` uses SoX for computing the spectrograms with 'lin' and 'log' parameters,
  while it uses Librosa for the mel spectrogram.

You can install SoX from [Matteo Spanio](https://github.com/matteospanio)'s [fork](https://github.com/matteospanio/sox-extended),
which enables computing spectrograms with logarithmic scale on the y-axis. The function is not supported natively by SoX

Starting from the audio samples with the channels already separated, follow these steps:

1. Execute script `src/wav2spec.py -i [audio_samples] -s [scale] -o [spec_dir]`. This will take the audio samples and compute a spectrogram for each one
   of them, using the specified scale. The possible scales are 'log', 'lin' and 'mel', and the frequencies extracted are 0-20k Hz

2. Execute script `src/divide.py -i [spec_dir] -s [scale] -o [halfs]`. The script will divide the spectrograms in two halfs, one labeled
   'c' and one labeled 'w'

3. Execute script `src/segment.py -c [halfs_c] -w [halfs_w] -o [out_folder]`. This will extract numerous fixed-size segments from the two halfs
   computed at the previous step, which will be used as the dataset for the models

- If you want to skip these steps and download the archived datasets directly, you can get them [here](https://drive.google.com/drive/folders/1-XSowWtwhLuJ3vkEJ-t8XaG1RTapbWNw?usp=sharing)

---

### TODO:

- Script to divide audio samples in two channels
- Automate dataset extraction steps
- Update `model_fit_colab` to use all three scales
- main.py breaks when input is mono, need to fix
