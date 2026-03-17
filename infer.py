import os
import argparse
import torch
import torchaudio
import scipy.io.wavfile as wavfile
import numpy as np

from model import CNNDenoisingAutoencoder

def denoise_audio(input_file, output_file, model_path='denoised_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model
    model = CNNDenoisingAutoencoder().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please run 'python main.py' first to train and save the model.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # 2. Load the input audio file using scipy
    sample_rate, audio_data = wavfile.read(input_file)
    
    # Normalize array and handle dimensions
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)
        
    waveform = torch.from_numpy(audio_data)
    
    # Convert stereo to mono if necessary
    if waveform.dim() > 1 and waveform.shape[1] > 1:
        waveform = torch.mean(waveform, dim=1)
        
    # Shape should be (1, T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.transpose(0, 1)

    # Resample to 8000 Hz if necessary (the model was trained on 8000 Hz)
    if sample_rate != 8000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        waveform = resampler(waveform)
        sample_rate = 8000
    
    # Ensure dimensions match expected model input (padding to multiple of 256 might help, but let's just use STFT)
    n_fft = 256
    hop_length = 64
    window = torch.hann_window(n_fft).to(device)

    waveform = waveform.to(device)
    expected_length = waveform.shape[1]

    # 3. Compute STFT
    stft = torch.stft(waveform.squeeze(0), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    noisy_mag = torch.abs(stft)
    noisy_phase = torch.angle(stft)
    
    # Needs dimensions (Batch, Channels, Freq, Time): (1, 1, F, T)
    noisy_mag_log = torch.log1p(noisy_mag).unsqueeze(0).unsqueeze(0)

    # Padding in time dimension if it's too short, or making divisible
    # The CNN architecture expects specific sizes to perfectly autoencode, 
    # but the interpolation in the forward pass of the model handles minor size discrepancies.
    
    print(f"Processing audio of length {expected_length / sample_rate:.2f} seconds...")
    
    # 4. Run inference
    with torch.no_grad():
        denoised_mag_log, mask = model(noisy_mag_log)

    # 5. Reconstruct audio
    denoised_mag = torch.expm1(denoised_mag_log.squeeze(0).squeeze(0)) # Inverse log1p
    denoised_mag = torch.clamp(denoised_mag, min=0.0)

    # Recombine with original noisy phase
    denoised_stft = denoised_mag * torch.exp(1j * noisy_phase)

    # Inverse STFT
    denoised_audio = torch.istft(denoised_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=expected_length)
    
    # 6. Save output audio
    audio_np = denoised_audio.cpu().numpy().reshape(-1, 1)
    wavfile.write(output_file, sample_rate, audio_np.astype(np.float32).flatten())
    print(f"Successfully saved denoised audio to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise an audio file using the trained CNN Autoencoder.")
    parser.add_argument("input_pos", nargs='?', type=str, default=None, help="Path to the noisy input .wav file (positional)")
    parser.add_argument("--input", type=str, default=None, help="Path to the noisy input .wav file (flag)")
    parser.add_argument("--output", type=str, default="output_denoised.wav", help="Path to save the denoised .wav file")
    parser.add_argument("--model", type=str, default="denoised_model.pth", help="Path to the trained model weights (.pth)")

    args = parser.parse_args()
    
    input_file = args.input if args.input is not None else args.input_pos
    if input_file is None:
        parser.error("You must provide an input file, either positionally or via --input.")
        
    denoise_audio(input_file, args.output, args.model)
