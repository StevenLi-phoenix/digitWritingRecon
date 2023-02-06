import numpy as np
import tensorflow as tf
import random
from PIL import Image
model = tf.keras.models.load_model('model')
index = random.randint(0,9)
img = Image.open(f"digits/{index}.jpg").convert('L')
img = img.resize((28, 28), Image.ANTIALIAS)
img = np.stack((img,), axis=-1) / 255.0
img = img.reshape(1, 28, 28)

print(index, model(img).numpy()[0])

