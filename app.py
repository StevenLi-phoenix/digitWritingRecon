import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, json, jsonify

import matplotlib.pyplot as plt

# load model
__checkpoint_dir = './model'          # 模型文件路径

model = tf.keras.models.load_model(__checkpoint_dir)
digits = dict([(i,"data:image/jpg;base64,"+base64.b64encode(open("digits/%s.jpg"%i,"rb").read()).decode()) for i in range(10)])
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():  # put application's code here
    return render_template("index.html",title='Home')

@app.route('/Recognize',methods=['POST'])
def recognize():
    data = json.loads(request.form.get('data'))
    imagedata = data["test_image"]
    imagedata = imagedata[22:]
    img = Image.open(BytesIO(base64.b64decode(imagedata))).convert('L')
    # img = img.resize((28, 28), Image.ANTIALIAS)
    img = img.resize((28, 28))
    img = np.stack((img,), axis=-1)/ 255.0
    img = 1 - img.reshape(1,28,28)
    img = img / np.max(img)

    # digit picture could be created client side to save network
    result = model(img)[0].numpy()
    pred_result = np.argpartition(result, -3)[-3:]
    ind1 = np.argsort(result[pred_result])
    pred_result = pred_result[ind1][::-1]


    info = dict()
    info['pred1_image'] = digits[pred_result[0]]
    info['pred1_accuracy'] = str('{:.2%}'.format(result[pred_result[0]])) + " " + str(pred_result[0])
    info['pred2_image'] = digits[pred_result[1]]
    info['pred2_accuracy'] = str('{:.2%}'.format(result[pred_result[1]])) + " " + str(pred_result[1])
    info['pred3_image'] =  digits[pred_result[2]]
    info['pred3_accuracy'] = str('{:.2%}'.format(result[pred_result[2]])) + " " + str(pred_result[2])
    return jsonify(info)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
