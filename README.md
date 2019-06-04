# IrisRecognition
Iris Recognition on CASIA Iris Image Database. Course project for GR5293 Applied Machine Learning for Image Analysis in Columbia.

## Environment
Python 2.7
Numpy 1.15.3
Scipy 1.1
scikit-learn 0.14
scikit-image 0.14
Opencv 3.4.2 (Using Opencv 3.3 has a very diferent result，so please use this version)
matplotlib 2.2.3
tabulate 0.8.2

## IrisRecognition.py
IrisRecognition contains main function.
### Import
import numpy as np
import cv2
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
import IrisMatching as IM
import PerformanceEvaluation as PE

###Variables
rootpath: relative path of image folder.
train_features: features of images in training set. 
train_classes: classes of images in training set.
test_features: features of images in test set.  
test_classes: classes of images in test set.
total_fmrs: total fmrs of these times
total_fnmrs: total fnmrs of these times
crr_mean: mean value of accuracy
crr_u: upper bound of 95% confidence interval of accuracy
crr_l: lower bound of 95% confidence interval of accuracy
fmrs_mean: mean value of fmr
fmrs_u: upper bound of 95% confidence interval of fmr
fmrs_l: lower bound of 95% confidence interval of fmr
fnmrs_mean: mean value of fnmr
fnmrs_u: upper bound of 95% confidence interval of fnmr
fnmrs_l: lower bound of 95% confidence interval of fnmr

### Output
table_CRR function for table 3 in Ma's paper and draw ROC curve
performance_evaluation function for fig.10
FM_FNM_table function for table 4
FMR_conf function for fig.13_a
FNMR_conf function for fig.13_b

### Logic:
1. We first read images from files. Then conduct image processing including Iris Localization, Iris Normalization, Image Enhancement.

2. Then conduct Feature Extraction.

3. Then use IrisMatching with three kinds of distance measure to compute accuracy. 

4. Then compute accuracy of reduced feature and we can output table 3 and fig.10. 

5. We find when feature numbers are 200 and using cosine similarity, it will have the best accuracy of 92%. So we use Bootsrap with this combination to compute confidence interval of accuracy, fmr and fnmr.Running 100 times will take about 1 to 2 hours.

6. Then we output table 4 and fig.13.

#### IrisLocalization
This function is to localize the pupil. The function preprocesses the eye image and then apply hough transform to detect both the circles of pupil and iris.
##### Parameters:
eye: the input image of the eye
##### Return:
 np.array(iris): coordinates of circle of iris which are centre coordinates and radius
 np.array([xp,yp,rp]): the coordinates of the circle of pupil which are centre coordinates and radius
##### Variables:
xp: x coordinate of the circle of pupil
yp: y coordinate of the circle of pupil
rp: radius of the pupil
xi: x coordinate of the circle of iris
yi: y coordinate of the circle of iris
ri: radius of the iris
##### Logic:
1. We use a bilateral filter to blur the image. Then we find an initial center following the way in Ma's paper way on blur image.
2. Then we take 120*120 window of the image. We find that image binarization will not improve the results, so we don't do that.After that we apply Gaussian blur to the window,then we use Hough Circles to find the pupil of the image. If the pupil center is to close to boundary, we will find it again in the whole image.
3. In finding iris,we refered to https://github.com/xs2315/iris_recognition. We slightly enlarge pupil radius.Then we apply the median blur to the copy of the image of the eye. Subsequently we use canny edge detection and hough circles to find the coordinates of the iris of the eye. We set iris radius range from rp+45 to 150. Then we pick the circle with most number of peaks as our iris circle. If the center of iris is too far from pupil, we change it to pupil center.

### IrisNormalization.py
def IrisNormalization(image,inner_circle,outer_circle ):
Input: img: image of the eye
          inner_circle: pupil: [x,y,r] where x is the x-coordinate of the center of pupil, y is the y-coordinate of the center of pupil, r is the radius of pupil
          outer_circle: iris: [x,y,r] where x is the x-coordinate of the center of iris, y is the y-coordinate of the center of iris, r is the radius of iris
output: A matrix with shape (64,512)

In this function, we unfold the ring between the outer circle and inner circle into a rectangle of size 64*512. We start from an angle of 0 degree and unfold the ring counterclockwise.


### ImageEnhancement.py
def ImageEnhancement(normalized_iris):
Input: normalized_iris: this is the result of the IrisNormalization function
output: ROI: region of interest

In this function, we use the method described in Li Ma’s paper, which he did local image equalization. Then we only take the top 48 rows of the matrix, since the bottom of the rectangle usually include some parts of eyelashes and eyelid. So the region of interest (ROI) is a matrix with shape 48*512.


### FeatureExtraction.py
def defined_filter(x,y,f):
Input: x, y, f(frequency)
Output: Modulating function
This function is given by Li Ma’s paper.

def gabor_filter(x,y, space_constant_x, space_constant_y, f):
input: x,y
         space_constant_x: given by the paper, which is 3 and 4 for the first filter and second filter
         space_constant_y: given by the paper which is 1.5
         f: frequency defined by the gabor function 
This function defines the Gabor filter function which is described in Li Ma’s paper. The parameters used in defined_filter and gabor_filter is refered to scikit-image’s built-in gabor_kernel function:https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_gabor.py#L16

def FeatureExtraction(roi):
input: roi: the result of ImageEnhancement function
output: vector of roi
In this function, we use defined_filter function and gabor_filter function to construct two filters(same as kernels) first, then we use convolution method to convolve two filters on the roi to get two filtered images. Then we calculate the mean and mean absolute deviation of every 8*8 blocks of roi. This is a method proposed by Li Ma. The reason we use mean absolute deviation rather than standard deviation is because in another Li Ma’s paper, 'Iris Recognition Based on Multichannel Gabor Filtering', the author said using mean absolute deviation will get a slightly better result. 

### PerformanceEvaluation.py
def performance_evaluation(train_features, train_classes, test_features, test_classes):
Input: train_features: feature vectors of train set
         train_classes: class labels of train set
         test_features: feature vectors of test set
         test_classes: class labels of train set
Output: a graph of dimensionality of feature vectors vs. correct recognition rate
In this function, by setting different numbers of features we use, we then call the function IrisMatchingRed, which can give us correct recognition rate(CRR) of using three different distances. Then we draw the graph using this three different distance. This is Fig.10 in Li Ma's paper.

def table_CRR(train_features, train_classes, test_features, test_classes):
input: train_features: feature vectors of train set
         train_classes: class labels of train set
         test_features: feature vectors of test set
         test_classes: class labels of train set
Output: a table that has CRR of three different distances using all features and reduced number of features
In this function, we call function IrisMatching to get CRR of three different distances using all features, then we call function IrisMatchingRed to get CRR of three distances using reduced features. This is Table 3 in Li Ma's paper.


def FM_FNM_table (train_features, train_classes, test_features, test_classes, 3, thresholds):
Input: train_features: the feature vector of train set
          train_classes: the class label of train set
          test_features: the feature vector of test set
          test_classes: the class label of train set
          3: cosine distance
          thresholds: threshold values
Output: a table of false match and false nonmatch rates with different threshold values

In this function, we call function IrisMatchingBootstrap to get total_fmrs and total_fnmrs first, then we call function drawROCBootstrap to get fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u, then we print out table filled with those values correspondingly. We use cosine distance for distance measurement. This is table 4 in Li Ma's paper.

def FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
Input: fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u
Output: the graph of 'FMR Confidence Interval'

def FNMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
Input: fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u
Output: the graph of 'FNMR Confidence Interval'

In this two functions, we draw the ROC curves with confidence intervals, which is Fig.13 in Li MA's paper.

## IrisMatching.py
IrisMatching is for features reduction and using nearest center classifier for classification.In this file, we also compute false match rate and false non match rate.
### Import
from scipy.spatial import distance
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
### Functions

#### selectTestSample
This function is to randomly selct test samples in Bootsrap.
##### Parameters:
test_features: features of images in test set. Shape:432*1536 
test_classes: classes of images in test set. Shape: 432*1
##### Return:
sample_features: features of test samples. Shape: 108*1536
sample_classes: classes of test samples. Shape: 108*1
##### Variables:
index: The indexes of test features we select. Shape: 108*1
##### Logic:
1. We first randomly select 108 integers from 0 to 431 with replacement. These are indexes of the test samples. (This procedure is a little different from paper. We directly select the indexes of the samples rather than indexes of classes, but I think it will get the similar result since it is also random selection with replacement)

2. According to indexes, we find features and classes of these test samples.

3. Return features and classes of test samples

#### calcTest
This function is to predict class of a test sample using nearest center classifier and distances between this sample and every training data.
##### Parameters:
train_features: features of images in training set.  
train_classes: classes of images in training set. 
test_sample: features of this test sample. 
test_class: class of this test sample. !!We don't use it in classification, just for convenience of computing fmr and fnmr.!!
dist: parameter denotes which distance it uses. 1 means l1 distance, 2 means l2 distances and 3 means cosine similarity.
##### Return:
sample_class: class of this sample predicted by classiferfeatures of training samples.
distsm: a set of distances between this sample and training data with the same class. 
distsn: a set of distances between this sample and training data with different classes. 
##### Variables:
dists: distances between this sample and all training data
offset: offset when computing distances
##### Logic:
1. Accoding to dist parameter, computing distances between this test sample and every training data. When computing l1 distance, using distance.cityblock function in Scipy.spatial. 
For l2, using distance.euclidean function, for cosine, using distance.cosine function(reference:https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
We considered about rotation, so we add an offset to compute distance. The real distance should be the minimum distance of every possible offsets. Then we add this distance to dists array.

2. If the test sample has the same class with this training data, we add this distance to distsm list, otherwise we add it to distsn list. 
For example, if test sample belongs to class 1, the training data also belongs to class 1, we will add distance between them in distsm list.
!!This process doesn't affect result of classifier, it is just for convenience of computing fmr and fnmr.!!

3. After computing all distances, we select the nearest distance of this test sample and the class of this nearest center is the sample's class.

#### IrisMatching
This function is to classify all test data,calculate the total accuray and distances of every pairs of training data and test data.
##### Parameters:
train_features: features of images in training set.  
train_classes: classes of images in training set. 
test_features: features of images in test set. 
test_classes: classes of images in test set. 
dist: parameter denotes which distance it uses. 1 means l1 distance, 2 means l2 distances and 3 means cosine similarity.
##### Return:
crr: total accuracy of classification
distancesm: a set of distances between all test data and training data with the same classes. 
distancesn: a set of distances between all test data and training data with different classes. 
##### Variables:
total: total number of test data, should be 432.0
num: number of correct classification
test_class: test class predicted by classifier
##### Logic:
1. Using calcTest function to compute predicted class and distances of every test data. If the result is right, add 1.0 to num.

2. Add distances to distancesm or distancesn. These two lists will be used to compute fmr and fnmr in the future.

3. Calculate the total accuracy of classification using num/total. 

#### IrisMatchingRed
This function is to reduce features and classify all test data using reduced features. I seperate this function from IrisMatching for saving time.
##### Parameters:
train_features: features of images in training set. 
train_classes: classes of images in training set. 
test_features: features of images in test set. 
test_classes: classes of images in test set. 
n: number of features after reduction
##### Return:
l1crr: accuracy of classification with l1 distance using reduced features.
l2crr: accuracy of classification with l2 distance using reduced features.
coscrr: accuracy of classification with cosine similarity using reduced features. 
##### Variables:
train_redfeatures: features of images in training set after reduction
test_redfeatures: features of images in test set after reduction
l1knn, l2knn, cosknn: knn classifier with l1, l2, or cosine as metric. Set k=1 so it will be same with nearest center classifier.
l1class, l2class, cosclass: classes of test set predicted by classifier
##### Logic:
1. Because feature reduction is time consuming, I just compute it once and use the reducted features to compute results of three distances. Because lda works when feature numbers are less than class numbers, 
I use Locally Linear Embedding method to reduct features after feature numbers bigger than 107(reference:https://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding). This method doesn't work when feature numbers are bigger than sample numbers. We use training data to fit it,
and there are 324 training data, so it only works when feature numbers are less than 323. So when parameter n is bigger than 323, this fuction will use original features.

2. After feature reduction, we do classification. Because I just want to use this function to select best feature number, we don't need to save distance data. I use KneighborClassifier function in sklearn to save time.

3. After classification, we compute accuray of every distance measure.

#### calcROC:
This function is to calculate fmr and fnmr for drawing ROC curve in the future.
##### Parameters:
distancesm: a list of distances between all test data and training data with the same classes. 
distancesn: a list of distances between all test data and training data with different classes. 
thresholds: a list of thresholds using for calculating pairs of fmr and fnmr. We consider it as distance, which means when distance is larger than threshold,
it will not match, otherwise it will match. 
##### Return:
fmrs: a list of fmrs according to thresholds
fnmrs: a list of fnmrs according to thresholds 
##### Variables:
numm: number of distances of pairs in the same class 
numn: number of distances of pairs in different classes
fm: number of false match 
fnm: number of false nonmatch

##### Logic:
1. For every threshold, comparing it with every distance of pairs in the same class. If the distance is bigger than threshold, it means this pair don't match, but actually
it should match because they are in the same class, so it is a false non match. Then fnm plus 1.

2. For every threshold, comparing it with every distance of pairs in different classes. If the distance is smaller than threshold, it is a false match. Then fm plus 1.

3. Using fnm/numm to compute fnmr and fm/numn to compute fmr, then add it to fmrs and fnmrs.

#### IrisMatchingBootsrap
This function is to realize Bootsrap. After calculate accuracy of original features and reduced features using three kinds of distance measure, we know using cosine similarity on 200 features
will get best accuray, so I use it in Bootsrap.
##### Parameters:
train_features: features of images in training set. 
train_classes: classes of images in training set. 
test_features: features of images in test set.  
test_classes: classes of images in test set. 
times: repeating times
##### Return:
total_fmrs: total fmrs of these times
total_fnmrs: total fnmrs of these times
crr_mean: mean value of accuracy
crr_u: upper bound of 95% confidence interval of accuracy
crr_l: lower bound of 95% confidence interval of accuracy
##### Variables:
train_redfeatures: features of images in training set after reduction
test_redfeatures: features of images in test set after reduction
l1knn, l2knn, cosknn: knn classifier with l1, l2, or cosine as metric. Set k=1 so it will be same with nearest center classifier.
l1class, l2class, cosclass: classes of test set predicted by classifier
##### Logic:
1. Using Locally Linear Embedding method to reduct features into 200 features

2. Using selectTestSample to randomly select 108 test samples 

3. Using IrisMatching with cosine similarity to calculate crr, fmrs and fnmrs. Add these values to total values lists.

4. Repeating 2,3,4 for times of parameter times 

5. Calculate mean accuracy and standard deviation of accuracy. After so many times iteration, we consider accuracy with normal distribution.
Using mean+1.96*standard deviation to estimate upper bound of 95% confidence interval of accuracy and mean-1.96*standard deviation to estimate lower bound.

#### calcROCBootsrap
This function is to calculate mean, upper bound and lower bound of fmr and fnmr in Bootsrap. 
##### Parameters:
fmrs: total fmrs calculated by IrisMatchingBootsrap
fnmrs: total fnmrs calculated by IrisMatchingBootsrap
##### Return:
fmrs_mean: mean value of fmr
fmrs_u: upper bound of 95% confidence interval of fmr
fmrs_l: lower bound of 95% confidence interval of fmr
fnmrs_mean: mean value of fnmr
fnmrs_u: upper bound of 95% confidence interval of fnmr
fnmrs_l: lower bound of 95% confidence interval of fnmr
##### Logic:
The logic is similar to compute these values of accuracy. The results will be used for drawing ROC curve with confidence interval.

### PerformanceEvaluation.py
def performance_evaluation(train_features, train_classes, test_features, test_classes):
Input: train_features: feature vectors of train set
         train_classes: class labels of train set
         test_features: feature vectors of test set
         test_classes: class labels of train set
Output: a graph of dimensionality of feature vectors vs. correct recognition rate
In this function, by setting different numbers of features we use, we then call the function IrisMatchingRed, which can give us correct recognition rate(CRR) of using three different distances. Then we draw the graph using this three different distance. This is Fig.10 in Li Ma's paper.

def table_CRR(train_features, train_classes, test_features, test_classes):
input: train_features: feature vectors of train set
         train_classes: class labels of train set
         test_features: feature vectors of test set
         test_classes: class labels of train set
Output: a table that has CRR of three different distances using all features and reduced number of features
In this function, we call function IrisMatching to get CRR of three different distances using all features, then we call function IrisMatchingRed to get CRR of three distances using reduced features. This is Table 3 in Li Ma's paper.


def FM_FNM_table (train_features, train_classes, test_features, test_classes, 3, thresholds):
Input: train_features: the feature vector of train set
          train_classes: the class label of train set
          test_features: the feature vector of test set
          test_classes: the class label of train set
          3: cosine distance
          thresholds: threshold values
Output: a table of false match and false nonmatch rates with different threshold values

In this function, we call function IrisMatchingBootstrap to get total_fmrs and total_fnmrs first, then we call function drawROCBootstrap to get fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u, then we print out table filled with those values correspondingly. We use cosine distance for distance measurement. This is table 4 in Li Ma's paper.

def FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
Input: fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u
Output: the graph of 'FMR Confidence Interval'

def FNMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
Input: fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u
Output: the graph of 'FNMR Confidence Interval'

In this two functions, we draw the ROC curves with confidence intervals, which is Fig.13 in Li MA's paper.

## Reference
Ma et al., Personal Identification Based on Iris Texture Analysis, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 25, NO. 12, DECEMBER 2003
