# **Traffic Sign Recognition** 

## DJ Traffic Sign Write Up

### The following write up explains the key aspects of the traffic sign classification which was implemented using LeNet based approach developed by Yann LeCun.

---

** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: #  (Image References)

[image1]:  ./examples/hist_train.png "Visualization"
[image2]:  ./examples/norm.png "Normalized"
[image3]:  ./examples/random_noise.jpg "Random Noise"
[image4]:  ./examples/1.png "Traffic Sign 1"
[image5]:  ./examples/2.png "Traffic Sign 2"
[image6]:  ./examples/3.png "Traffic Sign 3"
[image7]:  ./examples/4.png "Traffic Sign 4"
[image8]:  ./examples/5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Reflection on my "ever in works" T1-P2 commits

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes =  43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data labels are distributed across various label counts for the training data.

![Histogram][image1]

As we can see, there are some traffic signs which appear more times than others. This is to be expected since some signs such as speed limit signs are more common than say animal crossing signs.

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the data from range of (0,255) to (-1,1). Here is the example of traffic sign image before and after normalizing.

![Histogram][image2]

I experimented with gray scaling and decided to keep all 3 layers going into the first layer of CNN. However, I decided to generate additional data using rotation (took help of some other examples on the github to accomplish this) since I wanted to increase the dataset for training purposes. As we are aware, the accuracy of the model tends to improve with the size of the training data.  

All examples of my images are included in the python notebook.

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3	     	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 													|
| Convolution 5x5x12	      	| 1x1 stride,  outputs 10x10x24 				
| RELU	    |       									|
| Max pooling		| 2x2 stride, outputs 5x5x24 with dropouts       									|
| Flatten				| 5x5x24 to 600 outputs        									|
| Fully Connected		| Input 600 Output 240												|
| RELU and Dropouts						|												|
| Fully connected | Input 240 Outputs 120    
|
|RELU and Dropouts |
|
| Fully connected | Input 120 Outputs 84
|
| RELU and Dropouts |
|
| Fully connected | Inputs 84 Outputs 43|  


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used standard Adam optimizer for the optimization. I experimented with various hyper parameters as well as the model layers. I have included two models in my submission. One of the two provides a better accuracy. I set following hyper parameters:-

* EPOCHS = 20
* BATCH SIZE = 128
* LEARNING RATE = 5e-4

I found that with those settings I was able to get validation accuracy above 94% as needed for this assignment. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
EPOCH 20 ...
- Validation Accuracy = 96.4%
- Training Accuracy = 100%
- Test Accuracy = 95.368%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
- My first architecture tried was standard LeNet to make the model work.  The reason was to ensure that entire python notebook compiled properly. I discovered some nuances on making the training sequence work which was helpful for subsequent model planning. Specifically the point of shuffling at the beginning of the training start was where I was making an error initially.

* What were some problems with the initial architecture?
   - My validation accuracy needed was stuck at 91%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 - My initial architecture was underfitting. I adjusted my architecture by studying other similar architectures on git hub to get my validation accuracy higher than 94%. I was able to find multiple solutions which ensured this. 
* Which parameters were tuned? How were they adjusted and why?
  - The parameter tuning was done mainly by adjusting the Epochs and learning rate as well as drop out rate. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	* CNN works well with this problem due to following two main reasons:-
		* Translation Invariance: Ability to spot similar features across entire image segment using parameter sharing
		* Hierarchy based classification: Different layers of CNN get tuned to identify different features (e.g. line, edges or dark spots etc).
* What architecture was chosen?
	* I used modified LeNet architecture based on some of the other studies done on GitHub as well as 
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five traffic signs that I downloaded from the web:-

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some of these images are difficult to classify as the resize 
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


