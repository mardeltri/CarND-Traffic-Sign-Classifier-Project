# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/TrafficSigns.png "Traffic Signs"
[image2]: ./images/NewDataset_Test.png "Visualization"
[image3]: ./images/NewDataset_Train.png "Visualization"
[image4]: ./images/NewDataset_Valid.png "Visualization"
[image5]: ./images/OriginalDataset_Test.png "Visualization"
[image6]: ./images/OriginalDataset_Train.png "Visualization"
[image7]: ./images/OriginalDataset_Valid.png "Visualization"
[image8]: ./images/mysigns.png "Google maps Traffic Signs"
[image9]: ./images/grayscale.png "Grayscale"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/orcus25/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the nympy methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, in order to identify each class, one sign of each one is depicted in the following image.
![Sign classes][image1]
The next three bar charts show how many pictures of each sign there are.
![Test dataset][image2]
![Train dataset][image3]
![Valid dataset][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Several techniques were applied to the images:
* Grayscale. It was observed that the prediction was improved when converting from RGB to grayscale. It is clear that the neural network does not need RGB color information to identify the sign. Indeed, it confuses the network.
* Normalize. It is well known that normalizing the inputs helps whichever optimization algorithm.

Here is an example of a traffic sign image before and after grayscaling.
![Grayscale transformation][image9]

However, converting the images was not enough to improve accuracy since it reached a limit of 0.90. As suggested, the data set has been augmented in order to provide more information to the network.

Three transformation techniques were implemented:
* Translation: New pictures were created by translating the images a random quantity of pixels. For this purpose, the function warpAffine has been used.
* Perspective: Images were lightly distorted with the method getPerspectiveTransform.
* Rotation: Signs were rotated a random angle between -10 and 10 degrees.

Thus, the new dataset size is 89860. The next three bar charts show how many pictures of each sign there are.
![Test dataset][image5]
![Train dataset][image6]
![Valid dataset][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| input 32x32x1, outputs 28x28x6, stride 1x1, valid padding|
| RELU					|												|
| Max pooling	      	| 2x2 stride, input 28x28x6, outputs 14x14x6 	|
| Convolution 3x3     	| input 14x14x6, outputs 10x10x16, stride 1x1, valid padding|
| Max pooling	      	| 2x2 stride, input 10x10x25, outputs 16x16x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, input 10x10x16, outputs 5x5x16 	|
| Flatten			    | input 5x5x16, output 400      				|
| Fully connected		| input 400, output 120        					|
| Dropout 				| 50% probability   							|
| Fully connected		| input 120, output 84        					|
| RELU					|												|
| Dropout 				| 50% probability   							|
| Fully connected		| input 84, output 10        					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained by trial and error always taking into account aspects adquired during the lessons. With the following hyperparameters a reasonable accuracy was achieved:
* Epoch: 27
* Bach size: 156
* Keep probability (dropout): 50%
* Learning rate 0.001
* Optimizer: AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.986
* test set accuracy of 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet architecture was set initially since it works perfectly to identify characters.
* What were some problems with the initial architecture? The initial architecture was modified with two dropout layers in order to learn a redundant representation for everything, thus achieving a more robust neural network.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? Epoch and bach size were adjusted in order to improve accuracy. With a high number of epochs the model may be overfitted but with few ones it could be underfitted, so, an intermediate value was chosen.
* What are some of the important design choices and why were they chosen? Convolutional layers are really useful in this case since they look for details which helps to differentiate one picture from another, such as curves or colors.

It can be observed that the model works very well with the test accuracy with 95.5% of correct decisions. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


