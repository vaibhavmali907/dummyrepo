import tensorflow as tf
import numpy as np

unique_labels = ['dyed-lifted-polyps','dyed-resection-margins','esophagitis','normal-cecum',
                 'normal-pylorus','normal-z-line','polyps','ulcerative-colitis']
model = tf.keras.models.load_model("saved_model/model.h5")


def predict(img_path):  # mandatory: function name should be predict and it accepts a string which is image location
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(100,100,1))
    img = np.expand_dims(img, axis=0)
    img = img.reshape(-1,100,100,3)
    yhat = model.predict(img)
    yhat = np.array(yhat)
    indices = np.argmax(yhat, axis=1)
    scores = yhat[np.arange(len(yhat)), indices]
    predicted_categories = [unique_labels[i] for i in indices]
    category = predicted_categories[0]
    confidence = round(scores[0] * 100, 2)
    output = category + " (Confidence: " + str(confidence) + "%)"
    return output