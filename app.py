from flask import Flask, render_template, request
import tensorflow as tf

class_names = ['MildDemented',
               'ModerateDemented',
               'NonDemented',
               'VeryMildDemented']

app = Flask(__name__)

model = tf.keras.models.load_model('./model_al.h5')     

 # Create a function to load and prepare images
def load_and_prep_image(filename, image_shape=224, scale=True):

  # Read in the image
  img = tf.io.read_file(filename)

  # Decode image into tensor
  img = tf.io.decode_image(img, channels=3)

  # Resize the image
  img = tf.image.resize(img, [image_shape, image_shape])

  # Scale? Yes/No
  if scale:
    # rescale the image (get all values between 0 & 1)
    return img/255.
  else:
    return img # don't need to rescale images for EfficientNet models in TensorFlow

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
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
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