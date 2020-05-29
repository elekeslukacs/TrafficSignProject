# TrafficSignProject

Traffic Sign Recognition

This project is a simple approach to implement a traffic sign recognition system.

It uses classical image processing methods (color segmentation) to detect traffic signs in the image and CNNs to recognize the type of the traffic signs. The models are trained using the GTSRB data set. 

The focus was on the detection part and image processing methods, for the two CNNs inspiration comes from with small modifications: 
- Murtaza Hassan.
  ”Classify Traffic Sins using CNN”.
  https://www.murtazahassan.com/classify-traffic-signs-using-cnn/
  https://github.com/murtazahassan/OpenCV-Python-Tutorials-for-Beginners/tree/master/Advance/TrafficSignsCNN

- Sanket Doshi. ”Traffic Sign Detection using Convolutional NeuralNetwork” 
  https://towardsdatascience.com/traffic-sign-detection-using-convolutional-neural-network-660fb32fe90e
  
You can run the application by running test_solution.py, you can choose two modes, selecting images and using your webcam.
  
*Note: If you want to run test_solution.py it uses functions from utils.py and you need the download the pickle model from the link attached at the end of this file. You can comment the part where "model1.p" is loaded, but then you can use only the "Select images" function.
  
*Note 2: You need OpenCV and Keras (and TensorFlow) installed.
  
Link to another model ("model1.p"):
  
https://drive.google.com/open?id=1grbAD7yYPDo7KqAgFSnFle70yygWvqYO
