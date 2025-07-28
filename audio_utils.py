import librosa
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt

def get_formant_bands_linear(sample_rate=16000, n_fft=2048):

    five_bands = {}
    nyquist = sample_rate / 2.0
    band_hz_ranges = {
        'Sub-F1': (0, 200),
        'F1': (200, 1000),
        'F2': (800, 2500),
        'F3': (1500, 3500),
        'Supra-F3': (3500, nyquist)
    }
    
    linear_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    
    for name, (f_low, f_high) in band_hz_ranges.items():
        min_bin = np.argmin(np.abs(linear_freqs - f_low))
        max_bin = np.argmin(np.abs(linear_freqs - f_high))
        five_bands[name] = (min_bin, max_bin)
        
    return five_bands

def apply_thresholding(spec_db, threshold_val, formant_thresh=True, target_sr=16000, n_fft=2048):

    thresholded_spec = np.copy(spec_db)
    global_min = np.min(spec_db)

    if formant_thresh:
        bands = get_formant_bands_linear(sample_rate=target_sr, n_fft=n_fft)
        for name, (min_bin, max_bin) in bands.items():
            band = spec_db[min_bin:max_bin+1, :]
            if band.size == 0: continue
            
            band_max = np.max(band)
            threshold = band_max - threshold_val
            
            band_to_modify = thresholded_spec[min_bin:max_bin+1, :]
            band_to_modify[band_to_modify < threshold] = global_min
    else:
        global_max = np.max(spec_db)
        threshold = global_max - threshold_val
        thresholded_spec[thresholded_spec < threshold] = global_min

    return thresholded_spec

def audio_to_spectrogram(file_path, n_fft=2048, hop_length=1024, fixed_length=35, target_sr=16000, threshold_val=None, formant_thresh=False):

    y, sr = librosa.load(file_path, sr=target_sr)
    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    if threshold_val is not None:
        S_db = apply_thresholding(S_db, 
                                  threshold_val=threshold_val, 
                                  formant_thresh=formant_thresh, 
                                  target_sr=target_sr, 
                                  n_fft=n_fft)
    
    spec_tensor = torch.tensor(S_db, dtype=torch.float32)
    
    if spec_tensor.shape[1] < fixed_length:
        pad_len = fixed_length - spec_tensor.shape[1]
        spec_tensor = F.pad(spec_tensor, (0, pad_len))
    else:
        spec_tensor = spec_tensor[:, :fixed_length]
        
    return spec_tensor.unsqueeze(0)

def batch_audio_to_spectrogram(file_paths, **kwargs):
    return [audio_to_spectrogram(fp, **kwargs) for fp in file_paths]

def load_audio_dataset(real_folder_path, fake_folder_path, num_items_per_class=None, **kwargs):

    if not os.path.exists(real_folder_path):
        raise ValueError(f"Real folder not found: {real_folder_path}")
    if not os.path.exists(fake_folder_path):
        raise ValueError(f"Fake folder not found: {fake_folder_path}")

    # List all supported audio files from the provided paths
    real_files = [os.path.join(real_folder_path, f) for f in os.listdir(real_folder_path)
                  if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    fake_files = [os.path.join(fake_folder_path, f) for f in os.listdir(fake_folder_path)
                  if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]

    # Sample a subset of files if requested
    if num_items_per_class is not None:
        real_files = random.sample(real_files, min(num_items_per_class, len(real_files)))
        fake_files = random.sample(fake_files, min(num_items_per_class, len(fake_files)))

    print(f"Loading {len(real_files)} real and {len(fake_files)} fake audio files.")

    # Process files into spectrograms
    real_audio = batch_audio_to_spectrogram(real_files, **kwargs)
    fake_audio = batch_audio_to_spectrogram(fake_files, **kwargs)

    return real_audio, fake_audio

def plot_spectrogram_pairs(real_audio_list, fake_audio_list,
                           num_pairs=3, show_formants=False,
                           sample_rate=16000, n_fft=2048):
 
    five_bands = {}
    if show_formants:
        five_bands = get_formant_bands_linear(sample_rate=sample_rate, n_fft=n_fft)

    num_to_plot = min(num_pairs, len(real_audio_list), len(fake_audio_list))
    if num_to_plot == 0:
        print("No audio pairs to plot.")
        return

    fig, axes = plt.subplots(num_to_plot, 2, figsize=(17, 6 * num_to_plot), squeeze=False)

    for i in range(num_to_plot):
        ax1, ax2 = axes[i, 0], axes[i, 1]

        ax1.set_title(f'Real Audio (Pair {i+1})')
        im1 = ax1.imshow(real_audio_list[i].squeeze(), aspect='auto', origin='lower', cmap='magma')
        fig.colorbar(im1, ax=ax1, format='%+2.0f dB')
        ax1.set_xlabel("Time Frames")
        ax1.set_ylabel("Frequency Bins")

        ax2.set_title(f'Fake Audio (Pair {i+1})')
        im2 = ax2.imshow(fake_audio_list[i].squeeze(), aspect='auto', origin='lower', cmap='magma')
        fig.colorbar(im2, ax=ax2, format='%+2.0f dB')
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Frequency Bins") 

        if show_formants:
            band_names = list(five_bands.keys())
            for idx, name in enumerate(band_names):
                min_bin, max_bin = five_bands[name]
                if idx < len(band_names) - 1:
                    ax1.axhline(y=max_bin, color='cyan', linestyle=':', linewidth=1.5)
                    ax2.axhline(y=max_bin, color='cyan', linestyle=':', linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()