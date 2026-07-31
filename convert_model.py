import os

with open('models/model_int8.tflite', 'rb') as f:
    data = f.read()

with open('esp32_code/model_data.cc', 'w') as f:
    f.write('#include "model_data.h"\n\n')
    f.write('const unsigned char model_int8_tflite[] = {\n')
    for i in range(0, len(data), 12):
        chunk = data[i:min(i+12, len(data))]
        line = ', '.join(['0x{:02x}'.format(b) for b in chunk])
        f.write('    ' + line + ',\n')
    f.write('};\n\n')
    f.write('const unsigned int model_int8_tflite_len = {};\n'.format(len(data)))

print('model_data.cc generated! {} bytes'.format(len(data)))