import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.contrib.tensorrt as trt


# Clear any previous session.
tf.keras.backend.clear_session()

save_pb_dir = '.'
model_fname = 'mask_detector.model'


def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    print("Start reading graph...")
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(
            graph.as_graph_def())
    
        print("Traversing nodes to rename them")
        for node in graphdef_inf.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'Assign':
                    node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]
        
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir,
                             save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

model = load_model(model_fname)

session = tf.keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)

frozen_graph = freeze_graph(session.graph, session, [
                            out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

graph_io.write_graph(trt_graph, "./model/",
                     "trt_graph.pb", as_text=False)
