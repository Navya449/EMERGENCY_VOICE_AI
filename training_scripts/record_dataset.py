import sounddevice as sd
import soundfile as sf
import numpy as np
import os

fs = 16000
duration = 1.0
samples_per_word = 100
words = ['help', 'save_me', 'noise']

print("=" * 50)
print("EMERGENCY KWS - DATASET RECORDER")
print("=" * 50)
print(f"Sample rate: {fs} Hz")
print(f"Duration: {duration} second each")
print(f"Samples per word: {samples_per_word}")
print(f"Words: {words}")
print("\nSpeak CLEARLY and LOUDLY at normal distance.")
print("Press Ctrl+C to stop anytime.\n")

for word in words:
    print(f"\n{'='*50}")
    print(f"RECORDING: {word.upper()}")
    print(f"{'='*50}")
    
    folder = f'dataset/{word}'
    os.makedirs(folder, exist_ok=True)
    
    # Count existing files to resume from
    existing = len([f for f in os.listdir(folder) if f.endswith('.wav') and f.startswith('rec_')])
    print(f"Existing recordings: {existing}")
    
    for i in range(existing, samples_per_word):
        print(f"\n  [{word}] Sample {i+1}/{samples_per_word}")
        print(f"  Press Enter to record...")
        input()
        
        print(f"  Recording...")
        recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        
        rms = np.sqrt(np.mean(recording**2))
        max_val = np.abs(recording).max()
        
        print(f"  RMS: {rms:.6f} | Max: {max_val:.6f}")
        
        if rms < 0.001:
            print(f"  ⚠️ TOO QUIET! Re-record this one? (y/n)")
            redo = input().lower()
            if redo == 'y':
                continue
        
        filename = f'{folder}/rec_{i:03d}.wav'
        sf.write(filename, recording, fs)
        print(f"  ✅ Saved: {filename}")

print(f"\n{'='*50}")
print("RECORDING COMPLETE!")
print(f"{'='*50}")