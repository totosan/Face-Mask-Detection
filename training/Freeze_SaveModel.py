import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

_input_saved_model_dir = '/home/toto/Face-Mask-Detection/training/mask_detect_orig.pb'
_output_saved_model_dir = './model'

converter = trt.TrtGraphConverter(
    input_saved_model_dir=_input_saved_model_dir,
    max_workspace_size_bytes=(11<32),
    precision_mode="FP16",
    maximum_cached_engines=100)
converter.convert()
converter.save(_output_saved_model_dir)