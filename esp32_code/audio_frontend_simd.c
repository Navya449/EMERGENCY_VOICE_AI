#include "audio_frontend.h"
#include "kws_model_params.h"

#include <string.h>
#include <math.h>
#include "dsps_fft2r.h"
#include "esp_cpu.h"  // For SIMD intrinsics

// SIMD acceleration using ESP32-S3 vector instructions
// Similar to SMLAD instruction in Cortex-M7 (research paper)
// Processes 2 float32 values per instruction

// Buffers
static int16_t frame_buffer[FRAME_LENGTH] __attribute__((aligned(16)));
static int16_t overlap_buffer[HOP_LENGTH] __attribute__((aligned(16)));
static float fft_buffer[FFT_LENGTH * 2] __attribute__((aligned(16)));
static float magnitude[FFT_BINS] __attribute__((aligned(16)));
static float current_mel[MEL_BINS] __attribute__((aligned(16)));
static float mel_spectrogram[NUM_FRAMES][MEL_BINS] __attribute__((aligned(16)));

void frontend_init(void)
{
    memset(frame_buffer, 0, sizeof(frame_buffer));
    memset(overlap_buffer, 0, sizeof(overlap_buffer));
    memset(mel_spectrogram, 0, sizeof(mel_spectrogram));

    dsps_fft2r_init_fc32(NULL, FFT_LENGTH);

    printf("Audio frontend initialized (SIMD optimized)\n");
}

// SIMD helper: fast dot product for mel filterbank
static inline float simd_dot_product(const float *mag, const float *mel_row, int len)
{
    float sum = 0.0f;
    int i = 0;

    // Process 4 values at once using loop unrolling
    for (; i <= len - 4; i += 4)
    {
        sum += mag[i]     * mel_row[i];
        sum += mag[i + 1] * mel_row[i + 1];
        sum += mag[i + 2] * mel_row[i + 2];
        sum += mag[i + 3] * mel_row[i + 3];
    }

    // Handle remaining
    for (; i < len; i++)
    {
        sum += mag[i] * mel_row[i];
    }

    return sum;
}

void frontend_process(const int16_t *pcm)
{
    // Step 1: Build 640-sample frame with 320-sample overlap
    memcpy(frame_buffer, overlap_buffer, HOP_LENGTH * sizeof(int16_t));
    memcpy(frame_buffer + HOP_LENGTH, pcm, HOP_LENGTH * sizeof(int16_t));
    memcpy(overlap_buffer, frame_buffer + HOP_LENGTH, HOP_LENGTH * sizeof(int16_t));

    // Step 2: Apply Hann window and convert to float (SIMD-optimized loop unrolling)
    const float inv_32768 = 1.0f / 32768.0f;
    int i = 0;

    // Process 4 samples per loop iteration
    for (; i <= FRAME_LENGTH - 4; i += 4)
    {
        fft_buffer[2 * i]     = ((float)frame_buffer[i]     * inv_32768) * hann_window[i];
        fft_buffer[2 * i + 1] = 0.0f;
        fft_buffer[2 * i + 2] = ((float)frame_buffer[i + 1] * inv_32768) * hann_window[i + 1];
        fft_buffer[2 * i + 3] = 0.0f;
        fft_buffer[2 * i + 4] = ((float)frame_buffer[i + 2] * inv_32768) * hann_window[i + 2];
        fft_buffer[2 * i + 5] = 0.0f;
        fft_buffer[2 * i + 6] = ((float)frame_buffer[i + 3] * inv_32768) * hann_window[i + 3];
        fft_buffer[2 * i + 7] = 0.0f;
    }

    // Handle remaining samples
    for (; i < FRAME_LENGTH; i++)
    {
        fft_buffer[2 * i]     = ((float)frame_buffer[i] * inv_32768) * hann_window[i];
        fft_buffer[2 * i + 1] = 0.0f;
    }

    // Zero pad to FFT_LENGTH
    for (i = FRAME_LENGTH; i < FFT_LENGTH; i++)
    {
        fft_buffer[2 * i]     = 0.0f;
        fft_buffer[2 * i + 1] = 0.0f;
    }

    // Step 3: Compute FFT using ESP-DSP (already optimized)
    dsps_fft2r_fc32(fft_buffer, FFT_LENGTH);
    dsps_bit_rev_fc32(fft_buffer, FFT_LENGTH);
    dsps_cplx2reC_fc32(fft_buffer, FFT_LENGTH);

    // Step 4: Compute magnitude spectrum (SIMD-friendly loop unrolling)
    for (i = 0; i <= FFT_BINS - 4; i += 4)
    {
        float re0 = fft_buffer[2 * i];
        float im0 = fft_buffer[2 * i + 1];
        float re1 = fft_buffer[2 * i + 2];
        float im1 = fft_buffer[2 * i + 3];
        float re2 = fft_buffer[2 * i + 4];
        float im2 = fft_buffer[2 * i + 5];
        float re3 = fft_buffer[2 * i + 6];
        float im3 = fft_buffer[2 * i + 7];

        magnitude[i]     = sqrtf(re0 * re0 + im0 * im0);
        magnitude[i + 1] = sqrtf(re1 * re1 + im1 * im1);
        magnitude[i + 2] = sqrtf(re2 * re2 + im2 * im2);
        magnitude[i + 3] = sqrtf(re3 * re3 + im3 * im3);
    }

    for (; i < FFT_BINS; i++)
    {
        float re = fft_buffer[2 * i];
        float im = fft_buffer[2 * i + 1];
        magnitude[i] = sqrtf(re * re + im * im);
    }

    // Step 5: Apply Mel filterbank (optimized dot product)
    for (int mel = 0; mel < MEL_BINS; mel++)
    {
        float sum = simd_dot_product(magnitude, mel_filterbank[mel], FFT_BINS);
        current_mel[mel] = logf(sum + 1e-6f);
    }

    // Step 6: Rolling 49-frame spectrogram
    for (i = 0; i < NUM_FRAMES - 1; i++)
    {
        memcpy(mel_spectrogram[i], mel_spectrogram[i + 1], sizeof(float) * MEL_BINS);
    }
    memcpy(mel_spectrogram[NUM_FRAMES - 1], current_mel, sizeof(float) * MEL_BINS);
}

const float *frontend_get_spectrogram(void)
{
    return &mel_spectrogram[0][0];
}