#!/bin/bash


/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --shapes=\'input_1\':1x28x28 --saveEngine=./model_batch1_fp32.trt
#/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --shapes=\'input_1\':1x28x28 --fp16 --saveEngine=./model_batch1_fp16.trt

#/usr/src/tensorrt/bin/./trtexec --batch=8 --loadEngine=./model_batch1_fp32.trt
/usr/src/tensorrt/bin/./trtexec --explicitBatch --loadEngine=./model_batch1_fp32.trt
