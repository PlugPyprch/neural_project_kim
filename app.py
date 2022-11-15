from flask import Flask, render_template, request
# import tensorflow as tf
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

class_names = ['MildDemented',
               'ModerateDemented',
               'NonDemented',
               'VeryMildDemented']

app = Flask(__name__)

model = load_model('./model_al.h5')     

 # Create a function to load and prepare images
def load_and_prep_image(filename, image_shape=224, scale=True):
  img = image.load_img(filename, target_size=(image_shape, image_shape))
  img = np.asarray(img)
  return img/255.

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './static/images/' + imagefile.filename
    imagefile.save(image_path)
    imagine = r"images/" + imagefile.filename 

    img = load_and_prep_image(image_path, scale=False)
    pred_prob = model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred_prob.argmax()]

    final_prob = "{:.2f}".format(pred_prob.max()*100)
    # print(final_prob)
    # print(pred_class)

    # classification = '%s (%s%%)' % (pred_class, final_prob)
    classification = '%s' % (pred_class)
    print(classification)
    print(imagine)

    return render_template('index.html', prediction=classification, imagefilepath=imagine)

if __name__ == '__main__':
  app.run()
    # app.run(port=3000, debug=True)