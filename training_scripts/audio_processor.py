import numpy as np

class AudioProcessor:
    """
    This class must be IDENTICAL in behavior to the ESP32 C code.
    Every constant here will be exported to the ESP32.
    """
    
    def __init__(self, sample_rate=16000, frame_length_ms=40, hop_length_ms=20,
                 fft_length=1024, num_mel_bins=64, num_frames=49):
        
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)  # 640
        self.hop_length = int(sample_rate * hop_length_ms / 1000)      # 320
        self.fft_length = fft_length                                   # 1024
        self.num_mel_bins = num_mel_bins                               # 64
        self.num_frames = num_frames                                   # 49
        self.fft_bins = fft_length // 2 + 1                            # 513
        
        self.window = np.hanning(self.frame_length).astype(np.float32)
        self.mel_filterbank = self._create_mel_filterbank().astype(np.float32)
        
        print(f"AudioProcessor initialized:")
        print(f"  Frame length: {self.frame_length} samples ({frame_length_ms}ms)")
        print(f"  Hop length: {self.hop_length} samples ({hop_length_ms}ms)")
        print(f"  FFT length: {self.fft_length}")
        print(f"  Mel bins: {self.num_mel_bins}")
        print(f"  Output shape: ({self.num_frames}, {self.num_mel_bins})")
    
    def _hz_to_mel(self, hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def _mel_to_hz(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _create_mel_filterbank(self):
        low_freq = 0
        high_freq = self.sample_rate / 2
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, self.num_mel_bins + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_indices = np.floor((self.fft_length + 1) * hz_points / self.sample_rate).astype(int)
        filterbank = np.zeros((self.num_mel_bins, self.fft_bins), dtype=np.float32)
        for m in range(1, self.num_mel_bins + 1):
            f_m_minus = bin_indices[m - 1]
            f_m = bin_indices[m]
            f_m_plus = bin_indices[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return filterbank
    
    def _manual_stft(self, audio):
        required_length = (self.num_frames - 1) * self.hop_length + self.frame_length
        if len(audio) < required_length:
            audio = np.pad(audio, (0, required_length - len(audio)), mode='constant')
        mag_spectrogram = np.zeros((self.fft_bins, self.num_frames), dtype=np.float32)
        for frame_idx in range(self.num_frames):
            start = frame_idx * self.hop_length
            frame = audio[start:start + self.frame_length].astype(np.float32)
            windowed_frame = frame * self.window
            fft_result = np.fft.rfft(windowed_frame, n=self.fft_length)
            magnitude = np.abs(fft_result).astype(np.float32)
            mag_spectrogram[:, frame_idx] = magnitude
        return mag_spectrogram
    
    def compute_log_mel_spectrogram(self, audio):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.flatten()
        mag_spectrogram = self._manual_stft(audio)
        mel_spectrogram = np.dot(self.mel_filterbank, mag_spectrogram)
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)
        log_mel_spectrogram = log_mel_spectrogram.T
        return log_mel_spectrogram.astype(np.float32)
    
    def get_export_constants(self):
        return {
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'fft_length': self.fft_length,
            'num_mel_bins': self.num_mel_bins,
            'num_frames': self.num_frames,
            'fft_bins': self.fft_bins,
            'window': self.window,
            'mel_filterbank': self.mel_filterbank,
        }


if __name__ == "__main__":
    processor = AudioProcessor()
    test_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    result = processor.compute_log_mel_spectrogram(test_audio)
    print(f"Output shape: {result.shape}")
    print(f"Value range: [{result.min():.4f}, {result.max():.4f}]")
    print("AudioProcessor test passed!")