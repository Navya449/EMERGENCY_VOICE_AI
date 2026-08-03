#include <Arduino.h>
#include <driver/i2s.h>
#include "audio_frontend.h"
#include "kws_model_params.h"
#include "model_data.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#define I2S_BCK 4
#define I2S_WS  5
#define I2S_DIN 7
#define TENSOR_ARENA_SIZE (70 * 1024)
#define MIN_DETECTION_INTERVAL_MS 1000
#define DETECTION_THRESHOLD 0.3f

static const char* class_labels[] = {"help", "save_me", "noise"};
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

void i2s_init() {
    i2s_config_t config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = HOP_LENGTH,
        .use_apll = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_BCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN
    };
    i2s_driver_install(I2S_NUM_0, &config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
    Serial.println("I2S OK");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\nEmergency KWS - ESP32-S3");

    frontend_init();
    i2s_init();

    const tflite::Model *model = tflite::GetModel(model_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        return;
    }

    static tflite::AllOpsResolver resolver;
   static tflite::MicroInterpreter static_interp(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr, nullptr, nullptr);
    interpreter = &static_interp;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor alloc failed!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.printf("Model loaded. Arena: %d KB\n", TENSOR_ARENA_SIZE/1024);
}

void loop() {
    static int frames = 0;
    static int16_t pcm[HOP_LENGTH];
    static unsigned long last_detect = 0;
    
    int32_t raw[HOP_LENGTH];
    size_t bytes;
    i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes, 10);
    if (bytes > 0) {
        for (int i = 0; i < HOP_LENGTH; i++) pcm[i] = (int16_t)(raw[i] >> 16);
    }

    frontend_process(pcm);
    frames++;

    if (frames < NUM_FRAMES) { delay(20); return; }

    const float *mel = frontend_get_spectrogram();
    for (int i = 0; i < NUM_FRAMES * MEL_BINS; i++) {
        float norm = (mel[i] - TRAIN_MEAN) / TRAIN_STD;
        int32_t q = (int32_t)roundf(norm / MODEL_INPUT_SCALE) + MODEL_INPUT_ZERO_POINT;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        input->data.int8[i] = (int8_t)q;
    }

    if (interpreter->Invoke() != kTfLiteOk) { delay(20); return; }

    float max_conf = 0; int max_class = LABEL_NOISE;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float c = (output->data.int8[i] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
        if (c > max_conf) { max_conf = c; max_class = i; }
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
            Serial.printf(">>> DETECTED: %s (%.1f%%) <<<\n", class_labels[max_class], max_conf*100);
            last_detect = millis();
        }
    }
    delay(20);
}