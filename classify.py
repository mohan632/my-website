import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from skimage import transform

Models = ['Model EfficientNet.keras','Model VGG19.keras','Model CNN.keras']

def get_model(modelNo):
    model_path = './static/Models/' + Models[modelNo]
    model = load_model(model_path)
    return model


def predict(image_data,model):
    loaded_model = get_model(model)
    img = img_to_array(image_data)
    np_image = transform.resize(img, (224, 224, 3))
    image4 = np.expand_dims(np_image, axis=0)
    result__ = loaded_model.predict(image4)
    return result__
