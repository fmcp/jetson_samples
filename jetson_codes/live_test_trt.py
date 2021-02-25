import os
import copy
import argparse
import time
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

LABELS = ['0', '1']

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
graph = tf.Graph()
graph.as_default()
session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()

def load_with_converter(path, precision, batch_size):
	"""Loads a saved model using a TF-TRT converter, and returns the converter
	"""

	params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
	if precision == 'int8':
		precision_mode = trt.TrtPrecisionMode.INT8
	elif precision == 'fp16':
		precision_mode = trt.TrtPrecisionMode.FP16
	else:
		precision_mode = trt.TrtPrecisionMode.FP32

	params = params._replace(
		precision_mode=precision_mode,
		max_batch_size=batch_size,
		max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
		maximum_cached_engines=100,
		minimum_segment_size=3,
		allow_build_at_runtime=True
	)

	import pprint
	print("%" * 85)
	pprint.pprint(params)
	print("%" * 85)

	converter = trt.TrtGraphConverterV2(
		input_saved_model_dir=path,
		conversion_params=params,
	)

	return converter

def live_test(infer, output_tensor_name):
	cap = cv2.VideoCapture(0)
	live_data = True
	while (live_data):
		# Capture frame-by-frame
		ret, orig_frame = cap.read()
		frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (224, 224))
		frame = np.expand_dims(frame, axis=0)

		# Predict
		start = time.time()
		results = infer(tf.convert_to_tensor(frame, dtype=tf.dtypes.float32))[output_tensor_name].numpy()

		fps = 1.0 / (time.time() - start)
		results = np.squeeze(results)
		predicted_label = np.argmax(results)
		score = results[predicted_label]
		predicted_label = 0
		# Display the resulting frame
		res = "{} ({:0.1f}%); FPS: {:d}".format(LABELS[predicted_label], score*100, int(fps))
		cv2.putText(orig_frame, res, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
		cv2.imshow('Prediction', orig_frame)
		pressedKey = cv2.waitKey(1) & 0xFF
		if pressedKey == ord('q'):
			live_data = False

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# Input arguments
	parser = argparse.ArgumentParser()

	feature_parser = parser.add_mutually_exclusive_group(required=True)

	feature_parser.add_argument('--use_native_tensorflow', dest="use_tftrt", help="help", action='store_false')
	feature_parser.add_argument('--use_tftrt_model', dest="use_tftrt", action='store_true')

	parser.add_argument('--precision', dest="precision", type=str, default="fp16", choices=['int8', 'fp16', 'fp32'],
						help='Precision')
	parser.add_argument('--batch_size', dest="batch_size", type=int, default=1, help='Batch size')
	parser.add_argument('--model_path', dest="model_path", type=str, default="", help='Path to SavedModel')

	args = parser.parse_args()
	print("\n=========================================")
	print("Inference using: {} ...".format(
		"TF-TRT" if args.use_tftrt else "Native TensorFlow")
	)
	print("Batch size:", args.batch_size)
	if args.use_tftrt:
		print("Precision: ", args.precision)
	print("=========================================\n")
	model_path = args.model_path
	time.sleep(2)

	if args.use_tftrt:
		if not os.path.exists(os.path.join(model_path, "converted")):
			converter = load_with_converter(
				os.path.join(model_path),
				precision=args.precision,
				batch_size=args.batch_size
			)

			# fp16 or fp32
			xx = converter.convert()

			converter.save(
				os.path.join(model_path, "converted")
			)
		root = tf.saved_model.load(os.path.join(model_path, "converted"))
	else:
		root = tf.saved_model.load(model_path)

	infer = root.signatures['serving_default']
	output_tensorname = list(infer.structured_outputs.keys())[0]
	
	live_test(infer, output_tensorname)
