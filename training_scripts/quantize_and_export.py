import numpy as np
import tensorflow as tf
import pickle
import os
import json
from audio_processor import AudioProcessor

class ModelQuantizer:
    def __init__(self, model_path, data_path, processor):
        self.model_path = model_path
        self.data_path = data_path
        self.processor = processor
        
    def representative_dataset(self):
        X_train = np.load(os.path.join(self.data_path, 'X_train.npy'))
        X_train = X_train[..., np.newaxis]
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    
    def quantize_to_int8(self):
        print("\n" + "="*60)
        print("QUANTIZING MODEL TO INT8")
        print("="*60)
        
        # Load from .h5 (works better for quantization)
        model = tf.keras.models.load_model('../models/final_model.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_quant_model = converter.convert()
        with open('../models/model_int8.tflite', 'wb') as f:
            f.write(tflite_quant_model)
        print(f"INT8 model saved: ../models/model_int8.tflite")
        print(f"Size: {len(tflite_quant_model) / 1024:.1f} KB")
        return '../models/model_int8.tflite'
    
    def extract_model_info(self, tflite_path):
        print("\n" + "="*60)
        print("EXTRACTING MODEL INFORMATION")
        print("="*60)
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        model_info = {
            'input_scale': float(input_details['quantization_parameters']['scales'][0]),
            'input_zero_point': int(input_details['quantization_parameters']['zero_points'][0]),
            'output_scale': float(output_details['quantization_parameters']['scales'][0]),
            'output_zero_point': int(output_details['quantization_parameters']['zero_points'][0]),
        }
        print(f"Input scale: {model_info['input_scale']}")
        print(f"Input zero point: {model_info['input_zero_point']}")
        print(f"Output scale: {model_info['output_scale']}")
        print(f"Output zero point: {model_info['output_zero_point']}")
        return model_info
    
    def generate_esp32_header(self, model_info):
        print("\n" + "="*60)
        print("GENERATING ESP32 HEADER")
        print("="*60)
        constants = self.processor.get_export_constants()
        train_mean = float(np.load('../models/train_mean.npy')[0])
        train_std = float(np.load('../models/train_std.npy')[0])
        with open('../models/label_map.pkl', 'rb') as f:
            label_data = pickle.load(f)
        index_to_label = label_data['index_to_label']
        
        header = []
        header.append("#ifndef KWS_MODEL_PARAMS_H")
        header.append("#define KWS_MODEL_PARAMS_H")
        header.append("#include <stdint.h>")
        header.append("")
        header.append(f"#define FRAME_LENGTH {constants['frame_length']}")
        header.append(f"#define HOP_LENGTH {constants['hop_length']}")
        header.append(f"#define FFT_LENGTH {constants['fft_length']}")
        header.append(f"#define NUM_MEL_BINS {constants['num_mel_bins']}")
        header.append(f"#define NUM_FRAMES {constants['num_frames']}")
        header.append("")
        header.append(f"#define MODEL_INPUT_SCALE {model_info['input_scale']}f")
        header.append(f"#define MODEL_INPUT_ZERO_POINT {model_info['input_zero_point']}")
        header.append(f"#define MODEL_OUTPUT_SCALE {model_info['output_scale']}f")
        header.append(f"#define MODEL_OUTPUT_ZERO_POINT {model_info['output_zero_point']}")
        header.append(f"#define NUM_CLASSES 3")
        header.append("")
        header.append(f"#define TRAIN_MEAN {train_mean}f")
        header.append(f"#define TRAIN_STD {train_std}f")
        header.append("")
        header.append(f"#define LABEL_HELP 0")
        header.append(f"#define LABEL_SAVE_ME 1")
        header.append(f"#define LABEL_NOISE 2")
        header.append(f"#define DETECTION_THRESHOLD 0.8f")
        header.append("")
        header.append("#endif")
        
        os.makedirs('../esp32_code', exist_ok=True)
        with open('../esp32_code/kws_model_params.h', 'w') as f:
            f.write('\n'.join(header))
        print("Header saved: ../esp32_code/kws_model_params.h")


def main():
    processor = AudioProcessor()
    quantizer = ModelQuantizer('../models', '../models', processor)
    tflite_path = quantizer.quantize_to_int8()
    model_info = quantizer.extract_model_info(tflite_path)
    quantizer.generate_esp32_header(model_info)
    print("\nDONE! Files ready for ESP32:")
    print("  - models/model_int8.tflite")
    print("  - esp32_code/kws_model_params.h")


if __name__ == "__main__":
    main()