import librosa
import numpy as np
import torch
import torch.nn.functional as F
import os
import random

def audio_to_mel(file_path, n_fft=2048, hop_length=512, n_mels=128, fixed_length=65, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32)
    # Pad or truncate
    if mel_tensor.shape[1] < fixed_length:
        pad_len = fixed_length - mel_tensor.shape[1]
        mel_tensor = F.pad(mel_tensor, (0, pad_len))
    else:
        mel_tensor = mel_tensor[:, :fixed_length]
    return mel_tensor.unsqueeze(0)  # Shape: (1, n_mels, fixed_length)

def batch_audio_to_mel(file_paths, **kwargs):
    return [audio_to_mel(fp, **kwargs) for fp in file_paths]

def load_audio_dataset(folder_path, num_items_per_class=None, **kwargs):
    """
    Load audio files from real and fake folders and convert them to mel spectrograms.
    
    Args:
        folder_path (str): Path to the folder containing 'real' and 'fake' subfolders
        num_items_per_class (int, optional): Number of items to load from each class. 
                                           If None, loads all available files.
        **kwargs: Additional arguments to pass to audio_to_mel function
    
    Returns:
        tuple: (real_audio, fake_audio) where each is a list of mel spectrogram tensors
    """
    real_folder = os.path.join(folder_path, 'real')
    fake_folder = os.path.join(folder_path, 'fake')
    
    # Check if folders exist
    if not os.path.exists(real_folder):
        raise ValueError(f"Real folder not found: {real_folder}")
    if not os.path.exists(fake_folder):
        raise ValueError(f"Fake folder not found: {fake_folder}")
    
    # Get all audio files from each folder
    real_files = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
                  if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    fake_files = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
                  if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    
    # Limit the number of files if specified
    if num_items_per_class is not None:
        real_files = random.sample(real_files, min(num_items_per_class, len(real_files)))
        fake_files = random.sample(fake_files, min(num_items_per_class, len(fake_files)))
    
    print(f"Loading {len(real_files)} real audio files and {len(fake_files)} fake audio files")
    
    # Convert to mel spectrograms
    real_audio = batch_audio_to_mel(real_files, **kwargs)
    fake_audio = batch_audio_to_mel(fake_files, **kwargs)
    
    return real_audio, fake_audio
