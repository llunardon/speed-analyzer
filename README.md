# MPAI Audio Analyser: Speed Irregularities Detector

### Usage:

```
    python3 src/main.py "path_to_wav_file"
```

The script generates:

- log text file describing the irregularities found in each channel, with their relative timestamps
- a folder for each channel containing the respective .wav file and spectrogram
- a folder for each channel containing the respective segmented spectrogram

The output folders are only for manual revision, they can be eliminated in the final version of the program

### Dataset Extraction

1. The code used for extracting the spectrogram data from the audio files is in `notebooks/spectral_extraction.ipynb`
    - At the end of the notebook execution you'll have a dataset divided in two subfolders, one labeled 'correct' and the other 'wrong'

2. The code used for dividing the dataset in training-validation-testing and fitting the model is
in `notebooks/model_fit_colab.ipynb`, and it's suited for running in Google Colab

3. The colab notebook generates a zip archive of the model, which can be downloaded and used after extraction. The directory `models-def`
contains some pre-trained models