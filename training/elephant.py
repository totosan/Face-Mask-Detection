
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import numpy as np
import time

saved_model_dir_trt = 'mobilenet_v2.trt'


def get_elephant_x():
    img = tf.keras.preprocessing.image.load_img(
        'elephant.jpg', target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x

def time_my_model(model, data):
    times = []
    for i in range(20):
        start_time = time.time()
        one_prediction = model.predict(data)
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))


def time_trt_model():
    x = get_elephant_x()
    image_input = tf.constant(x.astype('float32'))
    times = []
    for i in range(20):
        start_time = time.time()
        global concrete_function
        one_prediction = concrete_func(input_1=image_input)
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))


def predict_elephant_std():
    #mobilenet_v2 = tf.keras.applications.MobileNetV2(weights='imagenet')
    #mobilenet_v2.save('mobilenet_v2', save_format="tf")
    mobilenet_v2 = tf.keras.models.load_model('mobilenet_v2')
    x = get_elephant_x()
    time_my_model(mobilenet_v2, x)


def convert_trt():
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode='FP16',
        is_dynamic_op=True)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir='mobilenet_v2',
        conversion_params=params)
    converter.convert()
    converter.save(saved_model_dir_trt)


def predict_elephant_trt():
    with tf.Session() as sess:
        root = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.SERVING],
            saved_model_dir_trt)

        inputs_mapping = dict(root.signature_def['serving_default'].inputs)
        outputs_mapping = dict(root.signature_def['serving_default'].outputs)

        print("inputMappings", inputs_mapping)
        print("outpuMappings", outputs_mapping)
        ops = sess.graph.get_operations()
        input_tensor = sess.graph.get_tensor_by_name('serving_default_input_1:0')
        output_tensor = sess.graph.get_tensor_by_name('PartitionedCall:0')
        print("input_tensor", input_tensor)
        print("output_tensor", output_tensor)
        x = get_elephant_x()
        
        output = sess.run(output_tensor,
                          feed_dict={input_tensor: x})
        
        return
        # Gather the ImageNet labels first and prepare them
        labels_path = tf.keras.utils.get_file(
            'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        imagenet_labels = np.array(open(labels_path).read().splitlines())

        # Perform inference
        labeling = concrete_func(tf.constant(x.astype('float32')))
        activations = tf.nn.softmax(labeling['predictions'])
        imagenet_labels[np.argsort(activations)[0, ::-1][:5]+1]


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    print('Using Tensorflow version: {0}'.format(tf.version.VERSION))
    print(tf.executing_eagerly())
    predict_elephant_trt()
