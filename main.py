# Importing all the libraries
from os import listdir
import os
import keras_vggface
import numpy as np
import tensorflow
from PIL import Image
from keras_vggface.utils import preprocess_input
from matplotlib import pyplot
from keras import Model
from keras_vggface.vggface import VGGFace
import mtcnn
from mtcnn import mtcnn
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import expand_dims
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import cosine

print(keras_vggface.__version__)
print(tensorflow.__version__)


#function for generating encodings of the images

def encodeimages(faces):
    encoding = []
    for i in faces:
        encoding.append(model.predict(i))
    return encoding


#Function for extracting the face from the image

def extract_face(filename, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = asarray(image)
    return face_array


# Directories where images are located
image_database = "images/"
test_database = "test/"

#Function for creating the array of the extracted face
def facearraycreator(database):
    faces = []
    dictionary = dict()
    for file in listdir(database):
        dirstring = str(database)
        person, extension = file.split(".")
        img = extract_face('%s%s.jpg' %(dirstring,person))
        # pyplot.figure()
        # pyplot.imshow(img)
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        faces.append(img)
        samples = np.asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        dictionary[person] = encodeimages(samples)
    return samples, dictionary


# Checking the scores on threshold of 0.5 by calculating the Cosine distance
def match(known_encoding, check_encoding, threshold = 0.5):
    score = cosine(known_encoding, check_encoding)
    if score <= threshold:
        print("Faces match : Score ==  %s", score)

    else:
        print("Faces do not match : Score == %s", score)
    return score


################################### MODEL ########################################
model = VGGFace(model='resnet50')
# Summarize input and output shape
print('Inputs: %s' % model.inputs)
print('Outputs: %s' % model.outputs)
#################################### MODEL ########################################




# Calling the function on images and test image
image_faces, real_database_dict = facearraycreator(image_database)
test_faces, test_database_dict = facearraycreator(test_database)


######################## TESTING ##################################################
test_scores = {}                    #dict to store the scores
for i in real_database_dict:
    p = match(real_database_dict[i][-1],list(test_database_dict.items())[-1][1])
    test_scores[i] = p

key_min = min(test_scores.keys(), key=(lambda k: test_scores[k]))
if test_scores[key_min] < 0.5 :     #below threshold and minimum
    print("Face Match ::: ", key_min)
    print("Match Score :: ", test_scores[key_min])
else:                               #above threshold not required
    print("No Match Found")
#####################################################################################