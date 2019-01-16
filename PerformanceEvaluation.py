from IrisMatching import IrisMatching, IrisMatchingRed, calcROC
from tabulate import tabulate #13min+
import matplotlib.pyplot as plt
import numpy as np

thresholds_2=[0.076,0.085,0.1]
def table_CRR(train_features, train_classes, test_features, test_classes):
    thresholds = np.arange(0.04,0.1,0.003)
    L1_1,_,_  = IrisMatching(train_features, train_classes, test_features, test_classes, 1)
    L2_1,_,_ = IrisMatching(train_features, train_classes, test_features, test_classes, 2)
    C_1,distsm,distsn = IrisMatching(train_features, train_classes, test_features, test_classes, 3)
    L1_2,L2_2,C_2=IrisMatchingRed(train_features, train_classes, test_features, test_classes, 200)
    print ("Correct recognition rate (%)")
    print tabulate([['L1 distance measure',L1_1*100 ,L1_2*100],['L2 distance measure', L2_1*100,L2_2*100], ['Cosine similarity measure', C_1*100,C_2*100]], headers=['Similartiy measure', 'Original feature set',"Reduced feature set"])
    fmrs, fnmrs = calcROC(distsm,distsn, thresholds)
    plt.plot(fmrs,fnmrs)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non_match Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.show()
    
#table_CRR(train_features, train_classes, test_features, test_classes)

def performance_evaluation(train_features, train_classes, test_features, test_classes):
    n = range(40, 201, 20)
    cos_crr=[]
    for i in range(len(n)):
        l1crr, l2crr, coscrr=IrisMatchingRed(train_features, train_classes, test_features, test_classes, n[i])
        cos_crr.append(coscrr*100)
    plt.plot(n,cos_crr,marker="*",color='navy')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('figure_10.png')
    plt.show()
#performance_evaluation(train_features, train_classes, test_features, test_classes)


def FM_FNM_table(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u,thresholds):
    print ("False Match and False Nonmatch Rates with Different Threshold Values")
    print tabulate([[thresholds[7], str(fmrs_mean[7])+"["+str(fmrs_l[7])+","+str(fmrs_u[7])+"]",str(fnmrs_mean[7])+"["+str(fnmrs_l[7])+","+str(fnmrs_u[7])+"]"], 
                    [thresholds[8], str(fmrs_mean[8])+"["+str(fmrs_l[8])+","+str(fmrs_u[8])+"]",str(fnmrs_mean[8])+"["+str(fnmrs_l[8])+","+str(fnmrs_u[8])+"]"],
                    [thresholds[9], str(fmrs_mean[9])+"["+str(fmrs_l[9])+","+str(fmrs_u[9])+"]",str(fnmrs_mean[9])+"["+str(fnmrs_l[9])+","+str(fnmrs_u[9])+"]"]],
                   headers=['Threshold', 'False match rate(%)',"False non-match rate(%)"])
#FM_FNM_table(train_features, train_classes, test_features, test_classes, thresholds_2)

def FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
    plt.figure()
    lw = 2
    plt.plot(fmrs_mean, fnmrs_mean, color='navy', lw=lw, linestyle='-')
    plt.plot(fmrs_l, fnmrs_mean, color='navy', lw=lw, linestyle='--')
    plt.plot(fmrs_u, fnmrs_mean, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 60])
    plt.ylim([0.0,40])
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non_match Rate(%)')
    plt.title('FMR Confidence Interval')
    plt.savefig('figure_13_a.png')
    plt.show()
    
def FNMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u):
    plt.figure()
    lw = 2
    plt.plot(fmrs_mean, fnmrs_mean, color='navy', lw=lw, linestyle='-')
    plt.plot(fmrs_mean, fnmrs_l, color='navy', lw=lw, linestyle='--')
    plt.plot(fmrs_mean, fnmrs_u, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 100])
    plt.ylim([0.0,40])
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non_match Rate(%)')
    plt.title('FNMR Confidence Interval')
    plt.savefig('figure_13_b.png')
    plt.show()

    
#FMR_conf(fmrs_mean,fmrs_l,fmrs_u,fnmrs_mean,fnmrs_l,fnmrs_u)
