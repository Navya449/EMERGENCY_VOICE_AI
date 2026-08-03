#ifndef AUDIO_FRONTEND_H
#define AUDIO_FRONTEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAMPLE_RATE    16000
#define FRAME_LENGTH   640
#define HOP_LENGTH     320
#define FFT_LENGTH     1024
#define FFT_BINS       513
#define MEL_BINS       64
#define NUM_FRAMES     49

void frontend_init(void);
void frontend_process(const int16_t *pcm);
const float *frontend_get_spectrogram(void);

#ifdef __cplusplus
}
#endif

#endif