
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
import numpy as np
import time
import os


saved_model_dir = 'mask_detector.model'
saved_model_dir_trt = saved_model_dir + '.trt'

# HELPER s


def get_Mask_x():
    img = tf.keras.preprocessing.image.load_img(
        './testimage/NoMask-training.jpg', target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x


def time_my_model(model, data):
    times = []
    
    predictor = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key='serving_default'
    )
    output = None
    for i in range(10):
        start_time = time.time()
        output = predictor({'input_1': data})
        delta = (time.time() - start_time)
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))
    print("Prediction output for the last instance:")
    for key in output.keys():
        print("{}: {}",key, output[key][0])

def time_trt_model():
    x = get_Mask_x()
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

# describing saved Model GraphDef


def describe_graph(graph_def, show_nodes=False):
    print('Input Feature Nodes: {}'.format(
        [node.name for node in graph_def.node if node.op == 'Placeholder']))
    print('')
    print('Unused Nodes: {}'.format(
        [node.name for node in graph_def.node if 'unused' in node.name]))
    print('')
    print('Output Nodes: {}'.format(
        [node.name for node in graph_def.node if (
            'predictions' in node.name or 'dense' in node.name)]))
    print('')
    print('Quantization Nodes: {}'.format(
        [node.name for node in graph_def.node if 'quant' in node.name]))
    print('')
    print('Constant Count: {}'.format(
        len([node for node in graph_def.node if node.op == 'Const'])))
    print('')
    print('Variable Count: {}'.format(
        len([node for node in graph_def.node if 'Variable' in node.op])))
    print('')
    print('Identity Count: {}'.format(
        len([node for node in graph_def.node if node.op == 'Identity'])))
    print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

    if show_nodes == True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))

# END Helpers ------------------------------------------------------------------


def get_inOut_of_stdModel():
    model = tf.keras.models.load_model(saved_model_dir)
    describe_graph(model.graph_def)


def predict_mask_std():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
        root = tf.saved_model.loader.load(
            sess,
            tags=[tf.saved_model.SERVING],
            export_dir=saved_model_dir)
        
        describe_graph(root.graph_def, False)
        #x = get_Mask_x()
        #time_my_model(root, x)


def freeze_model(saved_model_dir, output_node_names, output_filename):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ''
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags=tf.saved_model.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('graph freezed!')


def convert_trt():
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode='FP16',
        is_dynamic_op=True)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params)
    converter.convert()
    converter.save(saved_model_dir_trt)


# Loading the TensorRT Model
def predict_mask_trt():
    print("Do Config for GPU")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4    
    print("Do sess.run...")
    with tf.Session(config=config) as sess:
        root = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.SERVING],
            saved_model_dir_trt)

        describe_graph(root.graph_def, True)

        inputs_mapping = dict(root.signature_def['serving_default'].inputs)
        outputs_mapping = dict(root.signature_def['serving_default'].outputs)

        print("inputMappings", inputs_mapping)
        print("outpuMappings", outputs_mapping)
        
        ops = sess.graph.get_operations()
        input_tensor = sess.graph.get_tensor_by_name('serving_default_input_1:0')
        output_tensor = sess.graph.get_tensor_by_name('PartitionedCall:0')
        
        print("input_tensor", input_tensor)
        print("output_tensor", output_tensor)
        
        x = get_Mask_x()

        output = sess.run(output_tensor,
                          feed_dict={input_tensor: x})
        print(output)
        return


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    print('Using Tensorflow version: {0}'.format(tf.version.VERSION))
    print(' Executing eagerly:', tf.executing_eagerly())

    step = 3
    if step == 1:
        predict_mask_std()
    elif step==2:
        #freeze_model(saved_model_dir,"tensorflow/serving/predict",'frozen_model.pb')
        print('Converting to TRT')
        convert_trt()
    else:
        #mobilenet_v2 = tf.keras.models.load_model(saved_model_dir)
        predict_mask_trt()
