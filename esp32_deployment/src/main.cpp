#include <Arduino.h>
#include <driver/i2s.h>
#include "audio_frontend.h"
#include "kws_model_params.h"
#include "model_data.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ============================================================
// PIN DEFINITIONS - YOUR WIRING
// ============================================================
#define I2S0_BCK 4
#define I2S0_WS  5
#define I2S0_DIN 6

#define I2S1_BCK 15
#define I2S1_WS  16
#define I2S1_DIN 17

// ============================================================
// RING BUFFER CONFIGURATION
// ============================================================
#define RING_MS 500
#define RING_SAMPLES (SAMPLE_RATE * RING_MS / 1000)

static int16_t ring_buffer[4][RING_SAMPLES];
static volatile int ring_index = 0;

// ============================================================
// KWS CONFIGURATION
// ============================================================
#define TENSOR_ARENA_SIZE (70 * 1024)
#define MIN_DETECTION_INTERVAL_MS 1000

static const char* class_labels[] = {"help", "save_me", "noise"};
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

// ============================================================
// TDOA TASK HANDLE
// ============================================================
TaskHandle_t tdoa_task_handle = NULL;

// ============================================================
// I2S INITIALIZATION
// ============================================================
void init_i2s0() {
    i2s_config_t config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = HOP_LENGTH,
        .use_apll = false,
    };

    i2s_pin_config_t pins = {
        .bck_io_num = I2S0_BCK,
        .ws_io_num = I2S0_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S0_DIN
    };

    i2s_driver_install(I2S_NUM_0, &config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
}

void init_i2s1() {
    i2s_config_t config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = HOP_LENGTH,
        .use_apll = false,
    };

    i2s_pin_config_t pins = {
        .bck_io_num = I2S1_BCK,
        .ws_io_num = I2S1_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S1_DIN
    };

    i2s_driver_install(I2S_NUM_1, &config, 0, NULL);
    i2s_set_pin(I2S_NUM_1, &pins);
}

// ============================================================
// READ ALL 4 MICS
// ============================================================
void read_all_mics(int16_t *m1, int16_t *m2, int16_t *m3, int16_t *m4) {
    int32_t raw0[HOP_LENGTH * 2];
    size_t bytes0;
    i2s_read(I2S_NUM_0, raw0, sizeof(raw0), &bytes0, 10);

    if (bytes0 == sizeof(raw0)) {
        for (int i = 0; i < HOP_LENGTH; i++) {
            m1[i] = (int16_t)(raw0[i * 2] >> 16);
            m2[i] = (int16_t)(raw0[i * 2 + 1] >> 16);
        }
    } else {
        memset(m1, 0, HOP_LENGTH * sizeof(int16_t));
        memset(m2, 0, HOP_LENGTH * sizeof(int16_t));
    }

    int32_t raw1[HOP_LENGTH * 2];
    size_t bytes1;
    i2s_read(I2S_NUM_1, raw1, sizeof(raw1), &bytes1, 10);

    if (bytes1 == sizeof(raw1)) {
        for (int i = 0; i < HOP_LENGTH; i++) {
            m3[i] = (int16_t)(raw1[i * 2] >> 16);
            m4[i] = (int16_t)(raw1[i * 2 + 1] >> 16);
        }
    } else {
        memset(m3, 0, HOP_LENGTH * sizeof(int16_t));
        memset(m4, 0, HOP_LENGTH * sizeof(int16_t));
    }
}

// ============================================================
// STORE TO RING BUFFER
// ============================================================
void store_to_ring(int16_t *m1, int16_t *m2, int16_t *m3, int16_t *m4) {
    for (int i = 0; i < HOP_LENGTH; i++) {
        ring_buffer[0][ring_index] = m1[i];
        ring_buffer[1][ring_index] = m2[i];
        ring_buffer[2][ring_index] = m3[i];
        ring_buffer[3][ring_index] = m4[i];
        ring_index = (ring_index + 1) % RING_SAMPLES;
    }
}

// ============================================================
// TDOA: CROSS-CORRELATION (uses ring buffer directly)
// ============================================================
float compute_delay_ring(int mic_a, int mic_b) {
    float max_corr = 0;
    int best_delay = 0;
    int max_lag = 100;
    int len = RING_SAMPLES;

    for (int delay = -max_lag; delay <= max_lag; delay++) {
        float corr = 0;
        for (int i = max_lag; i < len - max_lag; i++) {
            int idx_a = (ring_index + i) % len;
            int idx_b = (ring_index + i + delay) % len;
            corr += (float)ring_buffer[mic_a][idx_a] * (float)ring_buffer[mic_b][idx_b];
        }
        if (corr > max_corr) {
            max_corr = corr;
            best_delay = delay;
        }
    }
    return (float)best_delay;
}

// ============================================================
// TDOA: ANGLE CALCULATION
// ============================================================
void calculate_angle(float delay_12, float delay_34, float *azimuth, float *elevation) {
    float mic_spacing = 0.05f;
    float sound_speed = 343.0f;
    float sample_period = 1.0f / SAMPLE_RATE;

    float time_12 = delay_12 * sample_period;
    float sin_az = (sound_speed * time_12) / mic_spacing;
    if (sin_az > 1.0f) sin_az = 1.0f;
    if (sin_az < -1.0f) sin_az = -1.0f;
    *azimuth = asinf(sin_az) * 180.0f / M_PI;

    float time_34 = delay_34 * sample_period;
    float sin_el = (sound_speed * time_34) / mic_spacing;
    if (sin_el > 1.0f) sin_el = 1.0f;
    if (sin_el < -1.0f) sin_el = -1.0f;
    *elevation = asinf(sin_el) * 180.0f / M_PI;
}

// ============================================================
// SEND TO DRONE (Serial output for now)
// ============================================================
void send_to_drone(float azimuth, float elevation) {
    Serial.printf("DRONE_CMD: TURN to Azimuth=%.1f Elevation=%.1f\n", azimuth, elevation);
}

// ============================================================
// TDOA TASK (Core 1)
// ============================================================
void tdoa_task(void *param) {
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        Serial.println("\n=== TDOA: Computing Direction ===");

        float delay_12 = compute_delay_ring(0, 1);
        float delay_34 = compute_delay_ring(2, 3);

        Serial.printf("Delay Mic1-Mic2: %.0f samples\n", delay_12);
        Serial.printf("Delay Mic3-Mic4: %.0f samples\n", delay_34);

        float azimuth = 0, elevation = 0;
        calculate_angle(delay_12, delay_34, &azimuth, &elevation);

        Serial.printf("Direction: Azimuth=%.1f  Elevation=%.1f\n", azimuth, elevation);
        send_to_drone(azimuth, elevation);

        Serial.println("=== TDOA: Done, sleeping ===\n");
    }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n=================================");
    Serial.println("Emergency KWS + TDOA - ESP32-S3");
    Serial.println("=================================");

    frontend_init();
    Serial.println("Audio frontend initialized");

    init_i2s0();
    Serial.println("I2S0 initialized (Mic1+Mic2 on GPIO 4,5,6)");

    init_i2s1();
    Serial.println("I2S1 initialized (Mic3+Mic4 on GPIO 15,16,17)");

    const tflite::Model *model = tflite::GetModel(model_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model version mismatch!");
        return;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interp(model, resolver, tensor_arena,
                                                   TENSOR_ARENA_SIZE, nullptr, nullptr, nullptr);
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: Tensor allocation failed!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.printf("KWS Model loaded (Arena: %d KB)\n", TENSOR_ARENA_SIZE / 1024);

    xTaskCreatePinnedToCore(
        tdoa_task,
        "TDOA",
        32768,
        NULL,
        2,
        &tdoa_task_handle,
        1
    );
    Serial.println("TDOA task created on Core 1");
    Serial.println("System ready. Listening...\n");
}

// ============================================================
// MAIN LOOP (Core 0)
// ============================================================
void loop() {
    static int frames_collected = 0;
    static int16_t pcm_m1[HOP_LENGTH], pcm_m2[HOP_LENGTH];
    static int16_t pcm_m3[HOP_LENGTH], pcm_m4[HOP_LENGTH];
    static unsigned long last_detect = 0;

    read_all_mics(pcm_m1, pcm_m2, pcm_m3, pcm_m4);
    store_to_ring(pcm_m1, pcm_m2, pcm_m3, pcm_m4);

    frontend_process(pcm_m1);
    frames_collected++;

    if (frames_collected < NUM_FRAMES) {
        delay(20);
        return;
    }

    const float *mel = frontend_get_spectrogram();
    for (int i = 0; i < NUM_FRAMES * MEL_BINS; i++) {
        float norm = (mel[i] - TRAIN_MEAN) / TRAIN_STD;
        int32_t q = (int32_t)roundf(norm / MODEL_INPUT_SCALE) + MODEL_INPUT_ZERO_POINT;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        input->data.int8[i] = (int8_t)q;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        delay(20);
        return;
    }

    float max_conf = 0;
    int max_class = LABEL_NOISE;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float c = (output->data.int8[i] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
        if (c > max_conf) {
            max_conf = c;
            max_class = i;
        }
    }

    static int cnt = 0;
    if (++cnt % 25 == 0) {
        Serial.printf("[KWS] help:%.1f%% save_me:%.1f%% noise:%.1f%%\n",
            (output->data.int8[LABEL_HELP] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE * 100,
            (output->data.int8[LABEL_SAVE_ME] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE * 100,
            (output->data.int8[LABEL_NOISE] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE * 100);
    }

    if (max_conf >= DETECTION_THRESHOLD && (millis() - last_detect) > MIN_DETECTION_INTERVAL_MS) {
        if (max_class == LABEL_HELP || max_class == LABEL_SAVE_ME) {
            Serial.printf(">>> DETECTED: %s (%.1f%%) <<<\n", class_labels[max_class], max_conf * 100);
            last_detect = millis();
            xTaskNotifyGive(tdoa_task_handle);
        }
    }

    delay(20);
}