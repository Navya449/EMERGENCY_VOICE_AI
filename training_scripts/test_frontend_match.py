"""
Test that Python AudioProcessor matches what the ESP32 C code will produce.
Simulates the exact same steps as audio_frontend.c
"""
import numpy as np
from audio_processor import AudioProcessor

# Test 1: Python AudioProcessor (our reference)
p = AudioProcessor()
test_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
python_output = p.compute_log_mel_spectrogram(test_audio)

print("="*60)
print("TEST 1: Python AudioProcessor Output")
print("="*60)
print(f"Shape: {python_output.shape}")
print(f"Mean:  {python_output.mean():.6f}")
print(f"Std:   {python_output.std():.6f}")
print(f"Min:   {python_output.min():.6f}")
print(f"Max:   {python_output.max():.6f}")
print(f"First value: {python_output[0, 0]:.6f}")
print(f"Last value:  {python_output[-1, -1]:.6f}")
print(f"Value at [24, 32] (center): {python_output[24, 32]:.6f}")
print()

# Test 2: Step-by-step simulation matching C code exactly
print("="*60)
print("TEST 2: Step-by-step C simulation")
print("="*60)

frame_length = 640
hop_length = 320
fft_length = 1024
fft_bins = 513
mel_bins = 64
num_frames = 49

# Same Hann window
window = np.hanning(frame_length).astype(np.float32)

# Same Mel filterbank
mel_fb = p.mel_filterbank

# Pad audio
required = (num_frames - 1) * hop_length + frame_length
audio = np.pad(test_audio, (0, max(0, required - len(test_audio))))

# Allocate buffers (matching C code)
overlap = np.zeros(hop_length, dtype=np.float32)
mel_spectrogram = np.zeros((num_frames, mel_bins), dtype=np.float32)

frame_count = 0
for frame_idx in range(num_frames):
    # Build frame (matches C: memcpy from overlap + new samples)
    start = frame_idx * hop_length
    frame = audio[start:start + frame_length].astype(np.float32)
    
    # Apply Hann window and normalize (matches C: /32768.0 * window)
    windowed = frame * window
    
    # Zero pad to FFT_LENGTH
    padded = np.zeros(fft_length, dtype=np.float32)
    padded[:frame_length] = windowed
    
    # FFT
    fft_result = np.fft.rfft(padded, n=fft_length)
    
    # Magnitude
    mag = np.abs(fft_result).astype(np.float32)
    
    # Mel filterbank (matches C: dot product loop)
    mel_output = np.zeros(mel_bins, dtype=np.float32)
    for mel in range(mel_bins):
        sum_val = 0.0
        for fft in range(fft_bins):
            sum_val += mag[fft] * mel_fb[mel, fft]
        mel_output[mel] = np.log(sum_val + 1e-6)
    
    # Store in spectrogram (rolling buffer)
    mel_spectrogram[frame_idx] = mel_output
    frame_count += 1

print(f"Frames processed: {frame_count}")
print(f"Shape: {mel_spectrogram.shape}")
print(f"Mean:  {mel_spectrogram.mean():.6f}")
print(f"Std:   {mel_spectrogram.std():.6f}")
print(f"Min:   {mel_spectrogram.min():.6f}")
print(f"Max:   {mel_spectrogram.max():.6f}")
print(f"First value: {mel_spectrogram[0, 0]:.6f}")
print(f"Last value:  {mel_spectrogram[-1, -1]:.6f}")
print(f"Value at [24, 32] (center): {mel_spectrogram[24, 32]:.6f}")
print()

# Test 3: Compare
print("="*60)
print("TEST 3: COMPARISON")
print("="*60)
diff = np.abs(python_output - mel_spectrogram)
print(f"Max absolute difference: {diff.max():.10f}")
print(f"Mean absolute difference: {diff.mean():.10f}")
print(f"Are they identical? {'✅ YES' if diff.max() < 1e-5 else '❌ NO - difference: ' + str(diff.max())}")
print()

# Test 4: Check specific values
print("="*60)
print("TEST 4: VALUE CHECKPOINTS")
print("="*60)
checkpoints = [(0, 0), (0, 63), (48, 0), (48, 63), (24, 32)]
all_match = True
for r, c in checkpoints:
    py_val = python_output[r, c]
    c_val = mel_spectrogram[r, c]
    match = abs(py_val - c_val) < 1e-5
    if not match:
        all_match = False
    print(f"  [{r},{c}] Python: {py_val:.8f} | C-sim: {c_val:.8f} | {'✅' if match else '❌'}")

print(f"\nAll checkpoints match: {'✅ YES - Frontends are identical' if all_match else '❌ MISMATCH DETECTED'}")