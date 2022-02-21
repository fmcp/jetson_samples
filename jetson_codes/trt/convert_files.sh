#!/bin/bash
python3 course_hdf5_pb.py --model_path ./model-final-pract2.hdf5 --model_path_save . --name_model course_jetson

python3 -m tf2onnx.convert --saved-model ./course_jetson --output ./course_jetson.onnx



