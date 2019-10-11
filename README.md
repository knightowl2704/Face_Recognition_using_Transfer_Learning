# Face_Recognition_using_Transfer_Learning
Implementing Face recognition using transfer learning using VGGFace (Not-Real time)

This repository shows how we can use transfer learning in Keras with the model **ResNet50** architecture. This repo is based on the keras-vggface by Refik Can Malli. It makes use of keras-vggface for importing various models from the vggface.models.py like RESNET50, VGG16, and SENET50. 

<br> 



Face Recognition can be broken down into 2 stages : 
* Face Detection 
* Face Verification

Haar Cascades can be used for face detection along with OpenCV for real time face detection. In this repo we are using MTCNN ( Multi-Task Cascaded Convolutional Neural Network) for detecting faces. 

Once the face is detected, we run through our pre trained ResNET model with the obtained image for finding out the encodings.
`model = model = VGGFace(model='resnet50')`

`encoding = model.predict(Image)`

Now that we know the encodings of the test image, we run through our database (Python Dictionary {person : encoding} ) to check for the closest encoding. This python dictionary consists of say 'n' number of people in the form of `{'piyush' : array[1.227 ....], 'person2': array[2.425....].... n }` etc.

For finding out the closest encoding, we can use simple Eucledian Distance or Cosine Distance. This can be achieved using the cosine() SciPy function. The maximum value of it can be 1.0 and minimum of 0. We pass in both the encodings of the test image and images in the database one after the other and calculate the score. 

`score = cosine(known_embedding, candidate_embedding)`

We define a threshold of 0.5 for determining whether the image is a match or not a match.


