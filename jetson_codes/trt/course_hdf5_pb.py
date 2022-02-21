import tensorflow as tf
import argparse

if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description='Course pb model')
    parser.add_argument('--model_path', type=str, default='', required=True, help="Path hdf5 or h5 model")
    parser.add_argument('--model_path_save', type=str, default='', required=True, help="Path to save model")
    parser.add_argument('--name_model', type=str, default='course_jetson', required=False, help="Name model")

    args = parser.parse_args()

    model_path = args.model_path
    name_model = args.name_model
    model_path_save = args.model_path_save

    model = tf.keras.models.load_model(model_path, compile=False)
    tf.saved_model.save(model, model_path_save + "/{}".format(name_model))
