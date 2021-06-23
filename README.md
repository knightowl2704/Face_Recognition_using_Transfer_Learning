# Face_Recognition_using_Transfer_Learning
Implementing Face recognition using transfer learning using VGGFace (Image + Real Time)

This repository shows implementation of transfer learning in Keras with the **ResNet50** architecture for face recognition system. This repo is based on the `keras-vggface` by Refik Can Malli. It uses keras-vggface for importing various models from the vggface's models.py, like **RESNET50**, **VGG16**, and **SENET50.** 

<br> 



Face Recognition consists of 2 stages : 
* Face Detection 
* Face Verification

Haar Cascades can be used for face detection along with OpenCV for real time face detection. In this repo I am using MTCNN ( Multi-Task Cascaded Convolutional Neural Network) for detecting faces. 

Once the face is detected, we run through our pre trained ResNET50 model with the obtained faceimage for finding out the encodings.

`model = VGGFace(model='resnet50')`

`encoding = model.predict(Image)`

Now that we know the encodings of the test image, we run through our database (Python Dictionary {person : encoding} ) to check for the closest encoding. This python dictionary consists of say 'n' number of people in the form of `{'piyush' : array[1.227 ....], 'person2': array[2.425....].... n }` etc.

For finding out the closest encoding, we can use simple Eucledian Distance or Cosine Distance. Both the functions are available in Scipy library and here I am using the cosine() function from SciPy. The maximum value of it can be 1.0 and minimum of 0. We pass in both the encodings of the test image and images in the database one after the other and calculate the score. 

`score = cosine(known_embedding, candidate_embedding)`

We define a threshold of 0.5 for determining whether the image is a match or not a match.

One test image is provided at a particular time to calculate the scores. Given below are some screenshots of the predictions.

![Test Image 1](https://github.com/knightowl2704/Face_Recognition_using_Transfer_Learning/blob/master/Screenshots/Screenshot%20(6).png)
![Test Image 2](https://github.com/knightowl2704/Face_Recognition_using_Transfer_Learning/blob/master/Screenshots/Screenshot%20(7).png)
![Test Image 3](https://github.com/knightowl2704/Face_Recognition_using_Transfer_Learning/blob/master/Screenshots/Screenshot%20(8).png)
![Test Image 4](https://github.com/knightowl2704/Face_Recognition_using_Transfer_Learning/blob/master/Screenshots/Screenshot%20(9).png)

The test image 1 shows a very high score because the label Selton Mello was included in the original training set of the pre-trained network. However the model still predicts on unknown test samples (test images 2,3,4) with considerable amount of accuracy, even when the model has not seen these inputs before this. Considering that the implementation does **not** involve any training at all the scores are decent.


<h1> Real Time Face Detection </h1>

Very similar to the Non-Real time face recognition, [this](https://github.com/knightowl2704/Face_Recognition_using_Transfer_Learning/blob/master/Real_time_implementation.py) makes use of **VGG_Face** architecture. The algorithm is same as before, however the face is now detected using **HaarCascade_frontal_face Classifier** for faster real-time response, a dictionary of employees is used to store the id and the representation (`representation = model.predict(employee_image)`), and each frame capture with **OpenCV** from webcam is checked for the employee id in the employee_dict for the CosineSimilarity of the representation of current frame and representation already saved in the employee_dict. Whenever the *cosineSimilarity* (`score = cosine(employee_representation, frame_representation)`) drops below the threshold, the name of employee is displayed on the frame. 


However for real time implementation, the model with pre trained weights performed with small accuracy as compared to Non-Real time implementation. Various models like **ResNet50, SeNet, and vgg16** were used for the same implementation and poor results were obtained.

The accuracy of real time model can be boosted by training some of the final layers of model with training dataset, due to very small dataset of images, this is not implemented yet. It will be added soon. Considering the fact that no training is done on the model whatsoever, the real time implementation performs well enough to detect and recognise 2-3 people simultaneously in one frame. Performance on a individual person is better, as compared to multiple face recognition.

For testing, just update the */images* folder with the employee images, with names in the format *xyz*.**jpg** (strictly) and the test image in test folder. 

For Real Time : Just update the */images* folder with the employee images, with names in the format *xyz*.**jpg** (strictly).
For Real Time, one may need to change the threshold value for obtaining optimum results, otherwise the model might predict only *Unknown* label on every face frame captured.


Since I did no training on for this implementation and used pre trained models, the default loss function for VGG16 is Eucledian Loss Function. 

`import keras.backend as K`


`def euclidean_distance_loss(y_true, y_pred):`
    `return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))`
    
