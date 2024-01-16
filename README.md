# MPAI Audio Analyzer: Speed Irregularities Detector

### Usage:

```
    python3 src/main.py -i [sample.wav] -s [scale] [opt_args]
```

### Optional Arguments:

- `-w`: resolution to use for the x-axis, in pixels/s. Default is 256px/s
- `-j`: length of the step to use for scanning of the spectrograms. Default is 64px

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

3. Execute script `src/segment.py -i [halfs] -o [out_folder] [opt_args]`. This will extract numerous fixed-size segments from the two halfs
   computed at the previous step, which will be used as the dataset for the models

   Optional arguments for `segment.py`:

   - `-w`: width of each segment, default is 256 pixels
   - `-j`: step used for the scanning of the spectrum. Default is 64 pixels

- If you want to skip these steps and download the archived datasets directly, you can get them [here](https://drive.google.com/file/d/1QI7oj-myHvzMUfvUid_h135LY8NxZGcC/view?usp=sharing)

---

### Test Dataset:

The dataset used for testing the models can be downloaded at this
[Google Drive link](https://drive.google.com/file/d/1gzg3pq3RKm9hRMZ5a8a_plY5_mm1iOfG/view?usp=sharing).

---

### TODO:

- Automate dataset extraction steps
- Update `model_fit_colab` to use all three scales
