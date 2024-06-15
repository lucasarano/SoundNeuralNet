# This file will cover the basics of creating a custom dataset for a nn model
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
# Note that Dataset, similar to nn, is a base class, which means that it is abstract.
# In other words, it needs to be inherited since it cannot be instantiated, just like java interfaces.
# Also this file will required to download UrbanSound8K to work, but I did not download it since it 6 gigabytes

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
    
    def __len__(self): # will return the number of samples in a dataset
        return len(self.annotations)

    def __getitem__(self, index): # similar to doing a_list[index], but a_list.__getitem__(index) instead
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}" # using iloc we can search the location of the desired data by its coordinate
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]