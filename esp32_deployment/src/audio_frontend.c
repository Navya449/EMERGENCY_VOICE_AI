#include "audio_frontend.h"
#include "kws_model_params.h"
#include <string.h>
#include <math.h>

static int16_t frame_buffer[FRAME_LENGTH] __attribute__((aligned(16)));
static int16_t overlap_buffer[HOP_LENGTH] __attribute__((aligned(16)));
static float fft_buffer[FFT_LENGTH * 2] __attribute__((aligned(16)));
static float magnitude[FFT_BINS] __attribute__((aligned(16)));
static float current_mel[MEL_BINS] __attribute__((aligned(16)));
static float mel_spectrogram[NUM_FRAMES][MEL_BINS] __attribute__((aligned(16)));

// Simple FFT (no external DSP needed)
static void simple_fft(float *data, int n) {
    int i, j, k;
    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2*i], ti = data[2*i+1];
            data[2*i] = data[2*j]; data[2*i+1] = data[2*j+1];
            data[2*j] = tr; data[2*j+1] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * 3.14159265f / len;
        float wpr = cosf(ang), wpi = sinf(ang);
        for (i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (j = 0; j < len/2; j++) {
                k = i + j;
                float tr = data[2*k], ti = data[2*k+1];
                float mr = data[2*(k+len/2)], mi = data[2*(k+len/2)+1];
                float re = wr * mr - wi * mi;
                float im = wr * mi + wi * mr;
                data[2*k] = tr + re; data[2*k+1] = ti + im;
                data[2*(k+len/2)] = tr - re; data[2*(k+len/2)+1] = ti - im;
                float tmp = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = tmp;
            }
        }
    }
}

void frontend_init(void) {
    memset(frame_buffer, 0, sizeof(frame_buffer));
    memset(overlap_buffer, 0, sizeof(overlap_buffer));
    memset(mel_spectrogram, 0, sizeof(mel_spectrogram));
}

void frontend_process(const int16_t *pcm) {
    memcpy(frame_buffer, overlap_buffer, HOP_LENGTH * sizeof(int16_t));
    memcpy(frame_buffer + HOP_LENGTH, pcm, HOP_LENGTH * sizeof(int16_t));
    memcpy(overlap_buffer, frame_buffer + HOP_LENGTH, HOP_LENGTH * sizeof(int16_t));

    for (int i = 0; i < FRAME_LENGTH; i++) {
        fft_buffer[2*i] = ((float)frame_buffer[i] / 32768.0f) * hann_window[i];
        fft_buffer[2*i+1] = 0.0f;
    }
    for (int i = FRAME_LENGTH; i < FFT_LENGTH; i++) {
        fft_buffer[2*i] = 0.0f; fft_buffer[2*i+1] = 0.0f;
    }

    simple_fft(fft_buffer, FFT_LENGTH);

    for (int i = 0; i < FFT_BINS; i++) {
        float re = fft_buffer[2*i], im = fft_buffer[2*i+1];
        magnitude[i] = sqrtf(re*re + im*im);
    }

    for (int mel = 0; mel < MEL_BINS; mel++) {
        float sum = 0.0f;
        for (int fft = 0; fft < FFT_BINS; fft++) {
            sum += magnitude[fft] * mel_filterbank[mel][fft];
        }
        current_mel[mel] = logf(sum + 1e-6f);
    }

    for (int i = 0; i < NUM_FRAMES - 1; i++) {
        memcpy(mel_spectrogram[i], mel_spectrogram[i+1], sizeof(float) * MEL_BINS);
    }
    memcpy(mel_spectrogram[NUM_FRAMES-1], current_mel, sizeof(float) * MEL_BINS);
}

const float *frontend_get_spectrogram(void) {
    return &mel_spectrogram[0][0];
}