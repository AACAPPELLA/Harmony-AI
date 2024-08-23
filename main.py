from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np

# 사용자 정의 레이어 정의
class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)

# 사용자 정의 KerasLayer 정의
class YamnetLayer(tf.keras.layers.Layer):
    def __init__(self, model_handle, **kwargs):
        super(YamnetLayer, self).__init__(**kwargs)
        self.model = hub.KerasLayer(model_handle, trainable=False)
        
    def call(self, inputs):
        return self.model(inputs)

# 모델 저장 경로 및 파일 이름
saved_model_path = './warning_prototype_model_mk3'

# 모델 로드
reloaded_model = tf.saved_model.load(saved_model_path)

# 실제 클래스 이름으로 대체
my_classes = ['fire_alert', 'air_alert', 'disaster_alert', 'warning_alert', 'dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
              'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane', 'mouse_click', 'pouring_water',
              'train', 'sheep', 'water_drops', 'church_bells', 'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
              'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects', 'laughing', 'hen',
              'engine', 'breathing', 'crying_baby', 'hand_saw', 'coughing', 'glass_breaking', 'snoring', 'toilet_flush', 'pig',
              'washing_machine', 'clock_tick', 'sneezing', 'rooster', 'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # 3MB

def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio, and normalize."""
    file_contents = tf.io.read_file(filename)
    
    # 디코딩, 단일 채널로 변환
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    
    # int64로 샘플 레이트 캐스팅
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    # 16kHz로 재샘플링
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    
    # 정규화 [-1.0, 1.0] 범위로
    wav = tf.cast(wav, tf.float32)
    max_val = tf.reduce_max(tf.abs(wav))
    wav = wav / max_val

    return wav

@app.route('/ai/predict', methods=['POST'])
def predict():
    try:
        # 요청에서 파일 가져옴
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        filepath = file.filename
        file.save(filepath)
        
        # WAV 파일 로드 및 정규화
        testing_wav_data = load_wav_16k_mono(filepath)
        
        # 텐서를 1D로 만듦
        testing_wav_data = tf.reshape(testing_wav_data, [-1])
        
        # np.array를 사용하여 배치 형태로 제공
        reloaded_results = reloaded_model(testing_wav_data, False, None)
        category_detect = my_classes[tf.math.argmax(reloaded_results).numpy()]
        
        is_warning = False
        for warning in ['fire_alert', 'air_alert', 'disaster_alert', 'warning_alert']:
            if warning in category_detect:
                is_warning = True
                break
        
        # 예측 결과 반환
        return jsonify({'is_warning': is_warning , 'prediction': category_detect})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
