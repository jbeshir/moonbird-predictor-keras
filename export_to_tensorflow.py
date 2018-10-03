import keras.backend as K
import keras.models
from sklearn.externals import joblib
import tensorflow.saved_model.builder as saved_model_builder
import tensorflow.saved_model.signature_constants as signature_constants
import tensorflow.saved_model.tag_constants as tag_constants
from tensorflow.saved_model.signature_def_utils import predict_signature_def

def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel.
    Copied from https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/keras/trainer/model.py"""

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

model = keras.models.load_model("model.hdf5")
to_savedmodel(model, 'SavedModel')

