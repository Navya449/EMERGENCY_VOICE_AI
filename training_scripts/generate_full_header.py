from audio_processor import AudioProcessor
import numpy as np
import json
import os

# Get project root (parent of training_scripts)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = AudioProcessor()
constants = p.get_export_constants()
train_mean = float(np.load(os.path.join(root, 'models', 'train_mean.npy'))[0])
train_std = float(np.load(os.path.join(root, 'models', 'train_std.npy'))[0])

with open(os.path.join(root, 'models', 'model_info.json')) as f:
    model_info = json.load(f)

lines = []
lines.append('#ifndef KWS_MODEL_PARAMS_H')
lines.append('#define KWS_MODEL_PARAMS_H')
lines.append('#include <stdint.h>')
lines.append('')
lines.append('// Audio parameters')
lines.append('#define SAMPLE_RATE 16000')
lines.append(f'#define FRAME_LENGTH {constants["frame_length"]}')
lines.append(f'#define HOP_LENGTH {constants["hop_length"]}')
lines.append(f'#define FFT_LENGTH {constants["fft_length"]}')
lines.append(f'#define NUM_MEL_BINS {constants["num_mel_bins"]}')
lines.append(f'#define NUM_FRAMES {constants["num_frames"]}')
lines.append(f'#define FFT_BINS {constants["fft_bins"]}')
lines.append('')
lines.append('// Model quantization')
lines.append(f'#define MODEL_INPUT_SCALE {model_info["input_scale"]}f')
lines.append(f'#define MODEL_INPUT_ZERO_POINT {model_info["input_zero_point"]}')
lines.append(f'#define MODEL_OUTPUT_SCALE {model_info["output_scale"]}f')
lines.append(f'#define MODEL_OUTPUT_ZERO_POINT {model_info["output_zero_point"]}')
lines.append('#define NUM_CLASSES 3')
lines.append('')
lines.append('// Normalization')
lines.append(f'#define TRAIN_MEAN {train_mean}f')
lines.append(f'#define TRAIN_STD {train_std}f')
lines.append('')
lines.append('// Labels')
lines.append('#define LABEL_HELP 0')
lines.append('#define LABEL_SAVE_ME 1')
lines.append('#define LABEL_NOISE 2')
lines.append('#define DETECTION_THRESHOLD 0.8f')
lines.append('')

lines.append('// Hann window')
lines.append(f'static const float hann_window[{constants["frame_length"]}] = {{')
w = constants['window']
for i in range(0, len(w), 8):
    chunk = w[i:i+8]
    lines.append('    ' + ', '.join([f'{v:.10f}f' for v in chunk]) + ',')
lines.append('};')
lines.append('')

lines.append('// Mel filterbank')
lines.append(f'static const float mel_filterbank[{constants["num_mel_bins"]}][{constants["fft_bins"]}] = {{')
mf = constants['mel_filterbank']
for i in range(len(mf)):
    row_parts = []
    for j in range(0, len(mf[i]), 8):
        chunk = mf[i][j:min(j+8, len(mf[i]))]
        row_parts.append(', '.join([f'{v:.10f}f' for v in chunk]))
    lines.append('    {' + ', '.join(row_parts) + '},')
lines.append('};')
lines.append('')
lines.append('#endif')

out_path = os.path.join(root, 'esp32_code', 'kws_model_params.h')
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f'Complete header written! {len(lines)} lines')
print(f'Saved to: {out_path}')
