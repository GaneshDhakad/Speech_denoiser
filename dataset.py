import os
import math
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class SpeechBlowerDataset(Dataset):
    def __init__(self, root_dir="./data", download=True, snr_db=0.0, segment_length=16000, num_samples=200):
        self.snr_db = snr_db
        self.segment_length = segment_length
        self.num_samples = num_samples
        self.use_synthetic = False
        self.data = []
        
        try:
            self.yesno = torchaudio.datasets.YESNO(root_dir, download=download)
            if len(self.yesno) == 0:
                self.use_synthetic = True
        except Exception as e:
            print(f"Failed to download YESNO dataset: {e}. Using synthetic speech-like data.")
            self.use_synthetic = True
            
        if self.use_synthetic:
            # Generate synthetic "speech" (formant-like sine waves with envelope)
            for _ in range(self.num_samples):
                t = torch.arange(segment_length).float() / 8000.0
                f0 = torch.randint(100, 300, (1,)).item()
                speech = torch.sin(2 * math.pi * f0 * t) + 0.5 * torch.sin(2 * math.pi * 2 * f0 * t) + 0.25 * torch.sin(2 * math.pi * 3 * f0 * t)
                envelope = torch.exp(-t * 2) * torch.sin(2 * math.pi * 2 * t)
                speech = speech * envelope
                speech = speech.unsqueeze(0) # (1, L)
                self.data.append(speech)

    def __len__(self):
        if self.use_synthetic:
            return self.num_samples
        return len(self.yesno)
    
    def __getitem__(self, idx):
        if self.use_synthetic:
            waveform = self.data[idx]
            sample_rate = 8000
        else:
            waveform, sample_rate, _ = self.yesno[idx]
            
        # Ensure waveform is segment_length
        if waveform.shape[1] > self.segment_length:
            start_idx = torch.randint(0, waveform.shape[1] - self.segment_length + 1, (1,)).item()
            clean_speech = waveform[:, start_idx:start_idx + self.segment_length]
        else:
            padding = self.segment_length - waveform.shape[1]
            clean_speech = torch.nn.functional.pad(waveform, (0, padding))
            
        # Normalize clean speech and ensure non-zero
        max_clean = torch.max(torch.abs(clean_speech))
        if max_clean > 0:
            clean_speech = clean_speech / max_clean
            
        # Generate blower noise: Brown noise + 50Hz hum
        white_noise = torch.randn_like(clean_speech)
        blower_noise = torch.cumsum(white_noise, dim=1)
        blower_noise = blower_noise - blower_noise.mean(dim=1, keepdim=True)
        
        t = torch.arange(self.segment_length).float() / sample_rate
        hum_50hz = torch.sin(2 * math.pi * 50 * t).unsqueeze(0)
        blower_noise = blower_noise + 2.0 * hum_50hz
        
        clean_rms = torch.sqrt(torch.mean(clean_speech**2))
        noise_rms = torch.sqrt(torch.mean(blower_noise**2))
        
        # Scale noise to target SNR
        target_noise_rms = clean_rms / (10 ** (self.snr_db / 20.0))
        
        if noise_rms > 0:
            blower_noise = blower_noise * (target_noise_rms / noise_rms)
            
        mixed_audio = clean_speech + blower_noise
        
        # Prevent clipping
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 0.99:
            mixed_audio = mixed_audio / (max_val + 1e-8)
            clean_speech = clean_speech / (max_val + 1e-8)
        
        # Precompute STFT
        n_fft = 256
        hop_length = 64
        window = torch.hann_window(n_fft)
        
        # STFT takes 1D input if we want BxFxT later without squeezed B
        clean_stft = torch.stft(clean_speech.squeeze(0), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mixed_stft = torch.stft(mixed_audio.squeeze(0), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        
        clean_mag = torch.abs(clean_stft)
        mixed_mag = torch.abs(mixed_stft)
        mixed_phase = torch.angle(mixed_stft)
        
        # Log1p scale for stabilizing neural net input
        clean_mag_log = torch.log1p(clean_mag)
        mixed_mag_log = torch.log1p(mixed_mag)
        
        return mixed_mag_log, clean_mag_log, mixed_phase, clean_speech, mixed_audio

def get_dataloader(batch_size=8, snr_db=0.0):
    dataset = SpeechBlowerDataset(snr_db=snr_db, num_samples=200) 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Drop last to avoid batch dimension mismatch if the dataset size isn't perfectly divisible
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader
