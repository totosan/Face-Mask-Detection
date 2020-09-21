import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

_input_saved_model_dir = './maysavedmodel'
_output_saved_model_dir = './model_saved'

converter = trt.TrtGraphConverter(
    input_saved_model_dir=_input_saved_model_dir,
    max_workspace_size_bytes=(11<32),
    precision_mode="FP16",
    maximum_cached_engines=100,
    max_batch_size=8)
converter.convert()
converter.save(_output_saved_model_dir)