import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np

from dataset import get_dataloader
from model import CNNDenoisingAutoencoder

def calculate_accuracy(clean_mag, noisy_mag, denoised_mag):
    # Accuracy based on magnitude spectrogram noise reduction
    error_before = torch.sum((clean_mag - noisy_mag) ** 2)
    error_after = torch.sum((clean_mag - denoised_mag) ** 2)
    accuracy = 1.0 - (error_after / (error_before + 1e-8))
    accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
    return accuracy.item() * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    batch_size = 8
    epochs = 40
    print("Initializing components...")
    train_loader, val_loader = get_dataloader(batch_size=batch_size, snr_db=5.0)
    
    # 2. Model, Loss, Optimizer
    model = CNNDenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    n_fft = 256
    hop_length = 64
    window = torch.hann_window(n_fft).to(device)
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for mixed_mag_log, clean_mag_log, mixed_phase, clean_speech, mixed_audio in train_loader:
            mixed_mag_log = mixed_mag_log.unsqueeze(1).to(device) # (B, 1, F, T)
            clean_mag_log = clean_mag_log.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            denoised_mag_log, mask = model(mixed_mag_log)
            
            loss = criterion(denoised_mag_log, clean_mag_log)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        best_clean = None
        best_denoised = None
        best_noisy = None
        with torch.no_grad():
            for mixed_mag_log, clean_mag_log, mixed_phase, clean_speech, mixed_audio in val_loader:
                mixed_mag_log = mixed_mag_log.unsqueeze(1).to(device)
                clean_mag_log = clean_mag_log.unsqueeze(1).to(device)
                mixed_phase = mixed_phase.to(device)
                clean_speech = clean_speech.to(device)
                
                denoised_mag_log, mask = model(mixed_mag_log)
                loss = criterion(denoised_mag_log, clean_mag_log)
                epoch_val_loss += loss.item()
                
                # Reconstruct audio
                denoised_mag = torch.expm1(denoised_mag_log.squeeze(1)) # Inverse log1p
                clean_mag = torch.expm1(clean_mag_log.squeeze(1))
                mixed_mag = torch.expm1(mixed_mag_log.squeeze(1))
                denoised_mag = torch.clamp(denoised_mag, min=0.0)
                
                # Polar to rectangular (complex)
                denoised_stft = denoised_mag * torch.exp(1j * mixed_phase)
                
                # ISTFT lengths should match original audio
                expected_length = clean_speech.shape[1]
                denoised_audio = torch.istft(denoised_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=expected_length)
                
                batch_acc = 0.0
                for b in range(denoised_audio.shape[0]):
                    acc = calculate_accuracy(clean_mag[b], mixed_mag[b], denoised_mag[b])
                    batch_acc += acc
                epoch_val_acc += batch_acc / denoised_audio.shape[0]
                
                if best_clean is None:
                    best_clean = clean_speech[0].cpu()
                    best_denoised = denoised_audio[0].cpu()
                    best_noisy = mixed_audio[0].cpu()
                
        epoch_val_loss /= len(val_loader)
        epoch_val_acc /= len(val_loader)
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2f}%")
        
    # Plotting Epochs vs Error
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Error (MSE Loss)')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Error (MSE Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Epochs vs Error for CNN Denoising Autoencoder')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(os.getcwd(), 'epochs_vs_error.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    
    # Save sample audio using scipy
    def save_wav(filename, tensor_audio):
        audio_np = tensor_audio.squeeze().numpy()
        audio_np = audio_np.reshape(-1, 1) # Scipy expects (N, 1) for mono if it checks shape[1], though (N,) should work in new versions. Let's make it (N,) but handle the error safely.
        wavfile.write(filename, 8000, audio_np.astype(np.float32).flatten())

    save_wav('sample_clean.wav', best_clean)
    save_wav('sample_noisy.wav', best_noisy)
    save_wav('sample_denoised.wav', best_denoised)
    print("Saved sample audio files for qualitative evaluation.")

    # Save the trained model weights
    model_path = os.path.join(os.getcwd(), 'denoised_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
