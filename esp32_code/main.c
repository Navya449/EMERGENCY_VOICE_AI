#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s.h"
#include "esp_log.h"

#include "audio_frontend.h"
#include "kws_model_params.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

static const char *TAG = "KWS";

// I2S configuration for INMP441
#define I2S_PORT    I2S_NUM_0
#define I2S_BCK     4
#define I2S_WS      5
#define I2S_DIN     6

// Tensor Arena (adjust based on model needs)
#define TENSOR_ARENA_SIZE (50 * 1024)

// Globals
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;

// Model data (from model_data.cc)
extern const unsigned char model_int8_tflite[];
extern const unsigned int model_int8_tflite_len;

// ---------- I2S Init ----------
void i2s_init(void)
{
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 320,
        .use_apll = false,
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DIN
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);

    ESP_LOGI(TAG, "I2S initialized");
}

// ---------- TFLite Init ----------
void tflite_init(void)
{
    const tflite::Model *model = tflite::GetModel(model_int8_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        return;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Tensor allocation failed!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "TFLite initialized");
    ESP_LOGI(TAG, "Input: scale=%.6f, zero_point=%d, shape=%dx%dx%d",
             input->params.scale, input->params.zero_point,
             input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Output: scale=%.6f, zero_point=%d, classes=%d",
             output->params.scale, output->params.zero_point, output->dims->data[1]);
}

// ---------- Main ----------
extern "C" void app_main(void)
{
    int32_t raw_samples[HOP_LENGTH];
    int16_t pcm[HOP_LENGTH];
    int frames_collected = 0;
    const char *last_detected = "none";
    int64_t last_detection_time = 0;

    frontend_init();
    i2s_init();
    tflite_init();

    ESP_LOGI(TAG, "Listening for: help, save_me");
    ESP_LOGI(TAG, "Detection threshold: %.2f", DETECTION_THRESHOLD);

    while (1)
    {
        // Read microphone
        size_t bytes_read;
        i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);

        // Convert 32-bit I2S to 16-bit PCM (INMP441 sends 24-bit in 32-bit slot)
        for (int i = 0; i < HOP_LENGTH; i++)
        {
            pcm[i] = (int16_t)(raw_samples[i] >> 16);
        }

        // Process audio frame
        frontend_process(pcm);
        frames_collected++;

        // Need 49 frames before first inference
        if (frames_collected < NUM_FRAMES)
        {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        // Get spectrogram and apply normalization
        const float *mel = frontend_get_spectrogram();

        // Normalize and quantize to INT8
        for (int i = 0; i < NUM_FRAMES * MEL_BINS; i++)
        {
            float normalized = (mel[i] - TRAIN_MEAN) / TRAIN_STD;
            int32_t quantized = (int32_t)roundf(normalized / MODEL_INPUT_SCALE) + MODEL_INPUT_ZERO_POINT;
            
            // Clamp to INT8 range
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            
            input->data.int8[i] = (int8_t)quantized;
        }

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk)
        {
            ESP_LOGE(TAG, "Inference failed!");
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        // Dequantize output and find max class
        float max_confidence = 0.0f;
        int max_class = LABEL_NOISE;

        for (int i = 0; i < NUM_CLASSES; i++)
        {
            float confidence = (output->data.int8[i] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
            
            if (confidence > max_confidence)
            {
                max_confidence = confidence;
                max_class = i;
            }
        }

        // Check detection threshold
        int64_t now = esp_timer_get_time() / 1000;  // milliseconds
        
        if (max_confidence >= DETECTION_THRESHOLD && 
            (now - last_detection_time) > MIN_DETECTION_INTERVAL_MS)
        {
            if (max_class == LABEL_HELP || max_class == LABEL_SAVE_ME)
            {
                ESP_LOGW(TAG, ">>> DETECTED: %s (confidence: %.2f%%) <<<", 
                         class_labels[max_class], max_confidence * 100);
                
                last_detected = class_labels[max_class];
                last_detection_time = now;

                // TODO: Trigger TDOA here
                // xTaskNotifyGive(tdoa_task_handle);
            }
        }

        // Debug output every 50 inferences
        static int infer_count = 0;
        infer_count++;
        if (infer_count % 50 == 0)
        {
            printf("[KWS] help: %.2f%% | save_me: %.2f%% | noise: %.2f%%\n",
                   (output->data.int8[LABEL_HELP] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE * 100,
                   (output->data.int8[LABEL_SAVE_ME] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE * 100,
                   (output->data.int8[LABEL_NOISE] - MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE * 100);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}