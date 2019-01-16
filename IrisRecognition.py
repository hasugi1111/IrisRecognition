#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
import IrisMatching as IM
import PerformanceEvaluation as PE
import datetime

rootpath = "./CASIA Iris Image Database (version 1.0)/"

train_features = np.zeros((324,1536))
train_classes = np.zeros(324, dtype = np.uint8)
test_features = np.zeros((432,1536))
test_classes = np.zeros(432, dtype = np.uint8)

starttime = datetime.datetime.now()

for i in xrange(1,109):
    filespath = rootpath + str(i).zfill(3)
    trainpath = filespath + "/1/"
    testpath = filespath + "/2/"
    for j in xrange(1,4):
        irispath = trainpath + str(i).zfill(3) + "_1_" + str(j) + ".bmp"
        img = cv2.imread(irispath, 0)
        iris, pupil = IrisLocalization(img)
        normalized = IrisNormalization(img, pupil, iris)
        ROI = ImageEnhancement(normalized)
        train_features[(i-1)*3+j-1, :] = FeatureExtraction(ROI)
        train_classes[(i-1)*3+j-1] = i
    for k in xrange(1,5):
        irispath = testpath + str(i).zfill(3) + "_2_" + str(k) + ".bmp"
        img = cv2.imread(irispath, 0)
        iris, pupil = IrisLocalization(img)
        normalized = IrisNormalization(img, pupil, iris)
        ROI = ImageEnhancement(normalized)
        test_features[(i-1)*4+k-1, :] = FeatureExtraction(ROI)
        test_classes[(i-1)*4+k-1] = i

endtime = datetime.datetime.now()

print 'image processing and feature extraction takes '+str((endtime-starttime).seconds)+' seconds'


PE.table_CRR(train_features, train_classes, test_features, test_classes)
PE.performance_evaluation(train_features, train_classes, test_features, test_classes)
#thresholds_2=[0.74,0.76,0.78]


# this part is for bootsrap
starttime = datetime.datetime.now() 
thresholds_3=np.arange(0.6,0.9,0.02)
times = 100 #running 100 times takes about 1 to 2 hours
total_fmrs, total_fnmrs, crr_mean, crr_u, crr_l = IM.IrisMatchingBootstrap(train_features, train_classes, test_features, test_classes,times,thresholds_3)
fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u = IM.calcROCBootstrap(total_fmrs, total_fnmrs)

endtime = datetime.datetime.now()

print 'Bootsrap takes'+str((endtime-starttime).seconds) + 'seconds'

fmrs_mean *= 100  #use for percent(%)
fmrs_l *= 100
fmrs_u *= 100
fnmrs_mean *= 100
fnmrs_l *= 100
fnmrs_u *= 100
PE.FM_FNM_table(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u, thresholds_3)
PE.FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u)
PE.FNMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u)
