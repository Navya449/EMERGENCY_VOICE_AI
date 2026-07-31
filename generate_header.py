import json
import numpy as np
import sys
sys.path.insert(0, 'training_scripts')
from audio_processor import AudioProcessor

model_info = {
    'input_scale': 0.019446276128292084,
    'input_zero_point': -7,
    'output_scale': 0.00390625,
    'output_zero_point': -128
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f)

p = AudioProcessor()
c = p.get_export_constants()
train_mean = float(np.load('models/train_mean.npy')[0])
train_std = float(np.load('models/train_std.npy')[0])

lines = []
lines.append('#ifndef KWS_MODEL_PARAMS_H')
lines.append('#define KWS_MODEL_PARAMS_H')
lines.append('#include <stdint.h>')
lines.append('')
lines.append('#define SAMPLE_RATE 16000')
lines.append('#define FRAME_LENGTH {}'.format(c['frame_length']))
lines.append('#define HOP_LENGTH {}'.format(c['hop_length']))
lines.append('#define FFT_LENGTH {}'.format(c['fft_length']))
lines.append('#define NUM_MEL_BINS {}'.format(c['num_mel_bins']))
lines.append('#define NUM_FRAMES {}'.format(c['num_frames']))
lines.append('#define FFT_BINS {}'.format(c['fft_bins']))
lines.append('')
lines.append('#define MODEL_INPUT_SCALE {}f'.format(model_info['input_scale']))
lines.append('#define MODEL_INPUT_ZERO_POINT {}'.format(model_info['input_zero_point']))
lines.append('#define MODEL_OUTPUT_SCALE {}f'.format(model_info['output_scale']))
lines.append('#define MODEL_OUTPUT_ZERO_POINT {}'.format(model_info['output_zero_point']))
lines.append('#define NUM_CLASSES 3')
lines.append('')
lines.append('#define TRAIN_MEAN {}f'.format(train_mean))
lines.append('#define TRAIN_STD {}f'.format(train_std))
lines.append('')
lines.append('#define LABEL_HELP 0')
lines.append('#define LABEL_SAVE_ME 1')
lines.append('#define LABEL_NOISE 2')
lines.append('#define DETECTION_THRESHOLD 0.8f')
lines.append('')
lines.append('static const float hann_window[640] = {')
w = c['window']
for i in range(0, 640, 8):
    vals = ', '.join(['{:.10f}f'.format(w[j]) for j in range(i, min(i+8, 640))])
    lines.append('    ' + vals + ',')
lines.append('};')
lines.append('')
lines.append('static const float mel_filterbank[64][513] = {')
mf = c['mel_filterbank']
for i in range(64):
    vals = ', '.join(['{:.10f}f'.format(mf[i][j]) for j in range(513)])
    lines.append('    {' + vals + '},')
lines.append('};')
lines.append('')
lines.append('#endif')

with open('esp32_code/kws_model_params.h', 'w') as f:
    f.write('\n'.join(lines))

print('Done! {} lines written.'.format(len(lines)))