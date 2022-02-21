#!/bin/bash

#FP32
/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --explicitBatch --saveEngine=./model_batch1_fp32.trt
/usr/src/tensorrt/bin/./trtexec --explicitBatch --loadEngine=./model_batch1_fp32.trt

#FP16
#/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --explicitBatch --saveEngine=./model_batch1_fp16.trt --fp16
#/usr/src/tensorrt/bin/./trtexec --explicitBatch --loadEngine=./model_batch1_fp16.trt --fp16

#INT8
#/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --explicitBatch --saveEngine=./model_batch1_int8.trt --int8
#/usr/src/tensorrt/bin/./trtexec --explicitBatch --loadEngine=./model_batch1_int8.trt --int8

#Best
#/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --explicitBatch --saveEngine=./model_batch1_best.trt --best
#/usr/src/tensorrt/bin/./trtexec --explicitBatch --loadEngine=./model_batch1_best.trt --best

#FP32 batch 8
#/usr/src/tensorrt/bin/./trtexec --onnx=./course_jetson.onnx --saveEngine=./model_batch8_fp32.trt --shapes=\'mobilenetv2_1_00_224_input:0\':8x224x224x3
#/usr/src/tensorrt/bin/./trtexec --explicitBatch --loadEngine=./model_batch8_fp32.trt

