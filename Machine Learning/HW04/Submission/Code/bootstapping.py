from run_me import run
import numpy as np
import random
import matplotlib.pyplot as plt

r = run()

def plotting(data, xlabel,title,figname):
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel("Count of Sample")
    plt.title(title)
    plt.show()
    plt.savefig(figname)

data1  = r.read_D1()
mean1 , variance1 = np.mean(data1) , np.var(data1)

data2 = r.read_D2()
mean2 , variance2 = np.mean(data2) , np.var(data2)

print("The mean of Dataset 1 : ", mean1)
print("Variance for D1:", variance1)
print("The mean of Dataset 2 : ", mean2)
print("Variance of D2:", variance2)
plotting(data1,"D1 dataset","Histogram for Dataset1","/Users/shivangisingh/Desktop/HW04/Figures/histogram_for_D1.png")
plotting(data2,"D2 dataset","Histogram for Dataset2","/Users/shivangisingh/Desktop/HW04/Figures/histogram_for_D2.png")



# Implement bootstrapping for the mean, using the percentile-based confidence interval method.
def percentile_based_bootstrapping(data, B, Dname):
    n = len(data)
    B_Means = []
    for b_sample in range(B):   
        sample = []
        for i in range(n):

            random_index = random.randint(0,n-1)
            sample.append(data[random_index])
        mean_ofsample = np.mean(sample)
        B_Means.append(mean_ofsample)
        bmean = np.mean(B_Means)
        bstandard_deviation = (np.var(B_Means))**(0.5)
    
    lbound = bmean - 1.96 *(bstandard_deviation)
    ubound = bmean + 1.96 *(bstandard_deviation)
    print("Bootsampling with ",B)
    print("The 95 percent confidence interval for ", Dname," is ",lbound," , " ,ubound)
    print("Variance of this Bootstrap sample is", np.var(B_Means), bmean)
    plotting(B_Means,"Bootstrap Sample Mean","Bootstrap Sample mean Count for Dataset 1, B=100","/Users/shivangisingh/Desktop/HW04/Figures/Bootstrapping(B=100,D1).png")


# percentile_based_bootstrapping(data1, 100, "Dataset 1")
# percentile_based_bootstrapping(data1, 1000, "Dataset 1")
# percentile_based_bootstrapping(data2, 100, "Dataset 2")
# percentile_based_bootstrapping(data2, 100, "Dataset 2")

###########################   Question 4   ###############################

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

# train_x, train_y = r.read_image_data()
# test_x, test_y = r.read_test_data()

# def calculate_accuracy(true, predicted):
#     total_correct = 0
#     for x,y in zip(true, predicted):
#         if x==y:
#             total_correct+=1
#     accuracy = total_correct/len(true)
#     print("Number of correct entries",accuracy)
#     return accuracy

# # Train Classifier Logistic Regression
# logis = LogisticRegression(C = 1 )
# logis.fit(train_x,train_y)
# lpredictions = logis.predict(test_x)

# lclassifier_accuracy = calculate_accuracy(test_y, lpredictions)
# print("Accuracy for the Logistic Regression Classifier", lclassifier_accuracy)

# # Train SVM
# clf = SVC(degree= 3, C = 10)
# clf.fit(train_x,train_y)
# spredictions = clf.predict(test_x)
# sclassifier_accuracy = calculate_accuracy(test_y, spredictions)
# print("Accuracy for the Support Vector Classifier", sclassifier_accuracy)



# def bootstrap_testing(clf1, clf2 , test_x , test_y, B ):
#     B_accuracy1 = []
#     B_accuracy2 = []
#     T_accuracy = []
#     n = len(test_x)
#     for b in range(B):
#         # Make your own testing
#         print("Running for ", b)
#         test_xs =[]
#         test_ys =[]
#         for i in range(n):
#             random_index = random.randint(0,n-1)
#             test_xs.append(test_x[random_index])
#             test_ys.append(test_y[random_index])

#         predictions1  = clf1.predict(test_xs)
#         predictions2  = clf2.predict(test_xs) 
#         accuracy1 = calculate_accuracy(test_ys, predictions1)
#         accuracy2 = calculate_accuracy(test_ys, predictions2)
#         B_accuracy1.append(accuracy1)
#         B_accuracy2.append(accuracy2)
#         T_accuracy.append(accuracy1-accuracy2)
#     bmean = np.mean(T_accuracy)
#     bstandard_deviation = (np.var(T_accuracy))**(0.5)
#     lbound = bmean - 1.96 *(bstandard_deviation)
#     ubound = bmean + 1.96 *(bstandard_deviation)
    
#     print("B",B,lbound,ubound)
#     plotting(T_accuracy,"T(b) Value","Frequency of T(b) values in Boostrapping")

# bootstrap_testing(logis,clf,test_x,test_y,100)