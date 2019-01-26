import keras.backend as K
import keras.models
from sklearn.externals import joblib
import tensorflow.saved_model.builder as saved_model_builder
import tensorflow.saved_model.signature_constants as signature_constants
import tensorflow.saved_model.tag_constants as tag_constants
from tensorflow.saved_model.signature_def_utils import predict_signature_def

import predictor


def to_savedmodel(basic_model, scaler, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel.
    Copied from https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/keras/trainer/model.py"""

    temp_weights = [layer.get_weights() for layer in basic_model.layers if len(layer.get_weights()) != 0]

    # Rebuild the model with our scaling baked into it, so we don't need to preprocess our input tensor.
    weighted_layer_i = 0
    model = predictor.buildmodel_defaults(scaler)
    for layer in model.layers:
        if len(layer.get_weights()) != 0:
            layer.set_weights(temp_weights[weighted_layer_i])
            weighted_layer_i += 1

    # Test prediction.
    print(predictor.predict(model, [0.9], None))

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
    builder.save()


if __name__ == '__main__':
    basic_model = keras.models.load_model("./model/model.hdf5")
    scaler = joblib.load("./model/scaler.save")

    to_savedmodel(basic_model, scaler, 'SavedModel')

