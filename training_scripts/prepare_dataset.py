import os
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
import pickle
from training_scripts.audio_processor import AudioProcessor

class DatasetPreparator:
    def __init__(self, dataset_path, processor=None):
        self.dataset_path = dataset_path
        self.processor = processor if processor else AudioProcessor()
        self.target_sr = 16000
        self.label_to_index = {}
        self.index_to_label = {}
        
    def verify_and_resample(self, file_path):
        try:
            audio, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            rms = np.sqrt(np.mean(audio ** 2))
            
            # Resample if needed (simple decimation or padding)
            if sr != self.target_sr:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * self.target_sr / sr))
            
            audio = audio.astype(np.float32)
            
            duration = len(audio) / self.target_sr
            
            target_length = self.target_sr  # 1 second
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
            
            return audio
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    
    def augment_audio(self, audio, num_augments=3):
        """Generate augmented versions. ONLY used for training set."""
        augmented = [audio]
        
        for i in range(num_augments):
            aug = audio.copy()
            
            # Add noise
            noise_level = np.random.uniform(0.001, 0.005)
            aug_noise = aug + np.random.normal(0, noise_level, len(aug))
            augmented.append(aug_noise.astype(np.float32))
            
            # Slight time shift
            shift = np.random.randint(-800, 800)
            aug_shift = np.roll(audio, shift)
            augmented.append(aug_shift.astype(np.float32))
            
            # Volume change
            vol_factor = np.random.uniform(0.8, 1.2)
            aug_vol = audio * vol_factor
            augmented.append(aug_vol.astype(np.float32))
        
        return augmented
    
    def load_dataset(self):
        print("\n" + "="*60)
        print("LOADING DATASET (3 Classes: help, save_me, noise)")
        print("="*60)
        
        class_folders = {
            'help': 0,
            'save_me': 1,
            'noise': 2
        }
        
        self.label_to_index = class_folders
        self.index_to_label = {v: k for k, v in class_folders.items()}
        
        all_audio = []
        all_labels = []
        
        for folder_name, label_idx in class_folders.items():
            folder_path = os.path.join(self.dataset_path, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"FOLDER NOT FOUND: {folder_path}")
                continue
            
            print(f"\nProcessing: {folder_name}")
            
            wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            
            valid_count = 0
            for wav_file in wav_files:
                file_path = os.path.join(folder_path, wav_file)
                audio = self.verify_and_resample(file_path)
                
                if audio is not None:
                    all_audio.append(audio)
                    all_labels.append(label_idx)
                    valid_count += 1
            
            print(f"   Loaded {valid_count}/{len(wav_files)} files")
        
        print(f"\nTotal valid samples: {len(all_audio)}")
        unique, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"   {self.index_to_label[label]}: {count} samples")
        
        return np.array(all_audio), np.array(all_labels)
    
    def compute_spectrograms(self, audio_array, labels_array):
        print("\n" + "="*60)
        print("COMPUTING SPECTROGRAMS")
        print("="*60)
        spectrograms = []
        for i, audio in enumerate(audio_array):
            if i % 50 == 0:
                print(f"  Processing {i+1}/{len(audio_array)}...")
            spec = self.processor.compute_log_mel_spectrogram(audio)
            spectrograms.append(spec)
        spectrograms = np.array(spectrograms, dtype=np.float32)
        print(f"Spectrogram array shape: {spectrograms.shape}")
        return spectrograms, labels_array
    
    def normalize_spectrograms(self, spectrograms):
        mean = np.mean(spectrograms)
        std = np.std(spectrograms)
        print(f"\nDataset statistics:")
        print(f"   Mean: {mean:.6f}")
        print(f"   Std:  {std:.6f}")
        normalized = (spectrograms - mean) / std
        return normalized, mean, std
    
    def prepare_and_save(self, output_dir='models'):
        audio, labels = self.load_dataset()
        
        if len(audio) == 0:
            print("\nNO VALID AUDIO FILES FOUND!")
            return None
        
        print("\n" + "="*60)
        print("SPLITTING DATASET (BEFORE AUGMENTATION)")
        print("="*60)
        
        indices = np.arange(len(audio))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])
        
        print(f"Training set:   {len(train_idx)} samples (will be augmented)")
        print(f"Validation set: {len(val_idx)} samples (original only)")
        print(f"Test set:       {len(test_idx)} samples (original only)")
        
        print("\n" + "="*60)
        print("AUGMENTING TRAINING SET ONLY")
        print("="*60)
        
        X_train_orig = audio[train_idx]
        y_train_orig = labels[train_idx]
        
        X_train_aug = []
        y_train_aug = []
        
        for audio_sample, label in zip(X_train_orig, y_train_orig):
            augmented_versions = self.augment_audio(audio_sample, num_augments=3)
            for aug_audio in augmented_versions:
                X_train_aug.append(aug_audio)
                y_train_aug.append(label)
        
        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)
        
        print(f"Original training samples: {len(X_train_orig)}")
        print(f"After augmentation: {len(X_train_aug)}")
        
        X_val = audio[val_idx]
        y_val = labels[val_idx]
        X_test = audio[test_idx]
        y_test = labels[test_idx]
        
        X_train_specs, y_train_final = self.compute_spectrograms(X_train_aug, y_train_aug)
        X_val_specs, y_val_final = self.compute_spectrograms(X_val, y_val)
        X_test_specs, y_test_final = self.compute_spectrograms(X_test, y_test)
        
        X_train_normalized, train_mean, train_std = self.normalize_spectrograms(X_train_specs)
        X_val_normalized = (X_val_specs - train_mean) / train_std
        X_test_normalized = (X_test_specs - train_mean) / train_std
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving to {output_dir}/")
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train_normalized)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train_final)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val_normalized)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val_final)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test_normalized)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test_final)
        np.save(os.path.join(output_dir, 'train_mean.npy'), np.array([train_mean]))
        np.save(os.path.join(output_dir, 'train_std.npy'), np.array([train_std]))
        
        with open(os.path.join(output_dir, 'label_map.pkl'), 'wb') as f:
            pickle.dump({
                'label_to_index': self.label_to_index,
                'index_to_label': self.index_to_label
            }, f)
        
        print(f"\n" + "="*60)
        print("DONE!")
        print("="*60)
        print(f"   Training samples:   {len(X_train_normalized)}")
        print(f"   Validation samples: {len(X_val_normalized)}")
        print(f"   Test samples:       {len(X_test_normalized)}")
        print(f"   Input shape: {X_train_normalized.shape[1:]}")
        
        return True


if __name__ == "__main__":
    processor = AudioProcessor()
    preparator = DatasetPreparator('dataset', processor)
    preparator.prepare_and_save('models')