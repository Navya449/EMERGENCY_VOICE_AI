#include "audio_frontend.h"
#include "kws_model_params.h"

#include <string.h>
#include <math.h>

// ESP-DSP for FFT
#include "dsps_fft2r.h"

// Buffers
static int16_t frame_buffer[FRAME_LENGTH];
static int16_t overlap_buffer[HOP_LENGTH];
static float fft_buffer[FFT_LENGTH * 2];
static float magnitude[FFT_BINS];
static float current_mel[MEL_BINS];
static float mel_spectrogram[NUM_FRAMES][MEL_BINS];

void frontend_init(void)
{
    memset(frame_buffer, 0, sizeof(frame_buffer));
    memset(overlap_buffer, 0, sizeof(overlap_buffer));
    memset(mel_spectrogram, 0, sizeof(mel_spectrogram));

    // Initialize ESP-DSP FFT
    dsps_fft2r_init_fc32(NULL, FFT_LENGTH);

    printf("Audio frontend initialized\n");
}

void frontend_process(const int16_t *pcm)
{
    // Step 1: Build 640-sample frame with 320-sample overlap
    memcpy(frame_buffer, overlap_buffer, HOP_LENGTH * sizeof(int16_t));
    memcpy(frame_buffer + HOP_LENGTH, pcm, HOP_LENGTH * sizeof(int16_t));
    memcpy(overlap_buffer, frame_buffer + HOP_LENGTH, HOP_LENGTH * sizeof(int16_t));

    // Step 2: Apply Hann window and convert to float
    for (int i = 0; i < FRAME_LENGTH; i++)
    {
        fft_buffer[2 * i]     = ((float)frame_buffer[i] / 32768.0f) * hann_window[i];
        fft_buffer[2 * i + 1] = 0.0f;
    }

    // Zero pad to FFT_LENGTH
    for (int i = FRAME_LENGTH; i < FFT_LENGTH; i++)
    {
        fft_buffer[2 * i]     = 0.0f;
        fft_buffer[2 * i + 1] = 0.0f;
    }

    // Step 3: Compute FFT
    dsps_fft2r_fc32(fft_buffer, FFT_LENGTH);
    dsps_bit_rev_fc32(fft_buffer, FFT_LENGTH);
    dsps_cplx2reC_fc32(fft_buffer, FFT_LENGTH);

    // Step 4: Compute magnitude spectrum
    for (int i = 0; i < FFT_BINS; i++)
    {
        float real = fft_buffer[2 * i];
        float imag = fft_buffer[2 * i + 1];
        magnitude[i] = sqrtf(real * real + imag * imag);
    }

    // Step 5: Apply Mel filterbank (matches Python: mel = spectrogram × mel_filterbank)
    for (int mel = 0; mel < MEL_BINS; mel++)
    {
        float sum = 0.0f;
        for (int fft = 0; fft < FFT_BINS; fft++)
        {
            sum += magnitude[fft] * mel_filterbank[mel][fft];
        }
        current_mel[mel] = logf(sum + 1e-6f);
    }

    // Step 6: Rolling 49-frame spectrogram (shift left, add new frame at end)
    for (int i = 0; i < NUM_FRAMES - 1; i++)
    {
        memcpy(mel_spectrogram[i], mel_spectrogram[i + 1], sizeof(float) * MEL_BINS);
    }
    memcpy(mel_spectrogram[NUM_FRAMES - 1], current_mel, sizeof(float) * MEL_BINS);
}

const float *frontend_get_spectrogram(void)
{
    return &mel_spectrogram[0][0];
}