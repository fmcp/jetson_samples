import argparse
import cv2
import os
import tensorflow as tf
import numpy as np
import time
import tensorrt as trt

LABELS = ["Banana", "Lemon", "Orange", "Strawberry"]  # UPDATE ME!!!!

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
graph = tf.Graph()
graph.as_default()
session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
TRT_LOGGER = trt.Logger()
import common


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 25, 60, 60, 1]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        # print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def live_test(model_path):
    #model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)
    live_data = True
    while (live_data):
        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (28, 28))
        frame = np.expand_dims(frame, axis=0)

        # Predict
        start = time.time()
        #results = model.predict(frame)

        with get_engine("no", model_path) as engine, engine.create_execution_context() as context:
            inputs_trt, outputs_trt, bindings_trt, stream_trt = common.allocate_buffers(engine, frame)
            # Do inference
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs_trt[0].host = frame
            # inputs[0].host = np.abs(inputs[0].host)
            results = common.do_inference_v2(context, bindings=bindings_trt, inputs=inputs_trt,
                                           outputs=outputs_trt, stream=stream_trt)

        fps = 1.0 / (time.time() - start)
        results = np.squeeze(results)
        predicted_label = np.argmax(results)
        score = results[predicted_label]

        # Display the resulting frame
        res = "{} ({:0.1f}%); FPS: {:d}".format(LABELS[predicted_label], score * 100, int(fps))
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
    parser = argparse.ArgumentParser(description='Test a CNN')
    parser.add_argument('--model_path', type=str, required=False, default="", help="Path to the trained model")

    args = parser.parse_args()
    model_path = args.model_path

    if os.path.exists(model_path):
        live_test(model_path)
    else:
        print("Wrong file_path!", model_path)
