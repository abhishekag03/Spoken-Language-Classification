#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import os
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from python_speech_features import mfcc
from python_speech_features.base import delta
import warnings
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import pickle as pk
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
# from sklearn.hmm import GaussianMixture


# # Pickle files

# In[174]:


def picklevar(filename, varname):
    file = open('./pickled\\' + filename, 'wb')
    pk.dump(varname, file)
    file.close()


# In[114]:


picklevar('clf_svm_C5gamma0.1.pkl',clf_svm)


# # Extract Audio Features

# In[6]:


def extract_features(audio_data, samplerate):
    mfcc_features = []
    hamming_window = numpy.hamming(400)
    if (len(audio_data.shape) > 1):
        audio_data = audio_data[:,0]
    for i in range(0,audio_data.shape[0]-400,240):
        trimmed = audio_data[i:i+400]
        hammed = numpy.multiply(hamming_window, trimmed)
        mfcced = mfcc(hammed,samplerate, nfft = 2048)
        mfcc_features.append(mfcced[0])
    delta_features = delta(mfcc_features, 1)
    return mfcc_features, delta_features


# # Testing

# In[ ]:


def testing(filename, clf):
    mfcc_features = []
    delta_features = []
    test_English = []
    #data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Eng_Sur.flac")
    data, samplerate = sf.read(filename)
    print(data.shape)
    print(samplerate)
    if(len(data.shape) > 1):
        data = data[:,0]
    mfcc_features, delta_features = extract_features(data, samplerate)
    test_English = numpy.concatenate((mfcc_features,delta_features),axis = 1)
    test_English = preprocessing.normalize(test_English)
    results = clf.predict(test_English)
    print(results)
    print(sum(results))
    print("English Accuracy:")
    print(1-sum(results)/len(test_English))
    print("Hindi Accuracy:")
    print(sum(results)/len(test_English))


# # Confusion Matrix

# In[ ]:


def confusion_matrix1(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# # ROC Curve

# In[232]:


def plot_roc_curve_multi_class(coordinates, labels, num_classes, classifier):
    fpr = dict()
    tpr = dict()
    area_under_curve = dict()
    y = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5])
    x_train, x_test, y_train, y_test = train_test_split(coordinates, y, test_size = 0.2)
    classifier = OneVsRestClassifier(classifier)
    classifier.fit(x_train, y_train)
    y_score = classifier.predict_proba(x_test)
#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     area_under_curve = auc(fpr, tpr)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        area_under_curve[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    area_under_curve["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    plt.figure()
    lw = 2
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f)' % area_under_curve[i])

        plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
#Reference taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html  


# In[ ]:





# In[8]:


# english_mfcc_features = []
# english_delta_features = []
# english_features = []
# num_iterations = 0
# for file in os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Me_English\\"):
#     audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Me_English\\" + file)
#     print(file, samplerate)
#     audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
#     english_mfcc_features.extend(audio_mfcc)
#     english_delta_features.extend(audio_delta)
#     num_iterations+=1


# In[10]:


# hindi_mfcc_features = []
# hindi_delta_features = []
# hindi_features = []
# num_iterations = 0
# for file in os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Me_Hindi\\"):
#     audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Me_Hindi\\" + file)
#     hamming_window = numpy.hamming(400)
#     print(file, samplerate)
#     audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
#     hindi_mfcc_features.extend(audio_mfcc)
#     hindi_delta_features.extend(audio_delta)
#     num_iterations+=1


# In[11]:


# print(len(hindi_mfcc_features), len(hindi_delta_features))
# print(len(english_mfcc_features), len(english_delta_features))
# if (len(hindi_mfcc_features) != len(hindi_delta_features)):
#     print("PROBLEM")


# In[12]:


# zeros = [0]*len(english_mfcc_features)
# english_mfcc_features = numpy.array(english_mfcc_features)
# english_delta_features = numpy.array(english_delta_features)
# x_English = numpy.concatenate((english_mfcc_features, english_delta_features), axis = 1)

# ones = [1]*len(hindi_mfcc_features)
# hindi_mfcc_features = numpy.array(hindi_mfcc_features)
# hindi_delta_features = numpy.array(hindi_delta_features)
# x_Hindi = numpy.concatenate((hindi_mfcc_features, hindi_delta_features),axis = 1)

# X = numpy.concatenate((x_English, x_Hindi), axis=0)
# Y = numpy.concatenate((zeros, ones), axis=0)
# normalized_X = preprocessing.normalize(X)


# # Classifiers

# # Feed Forward Neural Network Keras
# 

# In[145]:


X=x_train
Y=y_train
model = Sequential()
model.add(Dense(100, input_dim=26, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(6, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
# Fit the model
Y = to_categorical(Y)
model.fit(X, Y, epochs=40, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


# # Test Accuracy

# In[165]:


# kfold = KFold(n_splits=10, shuffle=True, random_state=7)
# results = cross_val_score(model, X, Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#print(model.predict(X),Y)

output_class = numpy.argmax(model.predict(X),axis=1)
true_class = numpy.argmax(Y,axis=1)
sum=0
for i in range(len(output_class)):
    if(output_class[i]==true_class[i]):
        sum+=1
print("Accuracy ", sum/len(output_class))
#print(numpy.argmax(model.predict(X),axis=1),numpy.argmax(Y,axis=1))


# In[168]:


Y_test=to_categorical(y_test)
output_class = numpy.argmax(model.predict(x_test),axis=1)
true_class = numpy.argmax(Y_test,axis=1)
sum=0
for i in range(len(output_class)):
    if(output_class[i]==true_class[i]):
        sum+=1
print("Accuracy ", sum/len(output_class))
#print(numpy.argmax(model.predict(X),axis=1),numpy.argmax(Y,axis=1))


# # SVM 

# In[146]:


#SVM
# parameters = [{'kernel': ['rbf'], 'gamma' : [0.001,0.05,0.1,10], 'C' : [0.001, 0.01, 0.1, 1,2.5,100]}]
# clf = GridSearchCV(svm.SVC(), parameters, cv = 5)
# clf = GridSearchCV(svm.SVC())
clf_svm = svm.SVC()
clf_svm.fit(x_train, y_train)


# # Logistic

# In[218]:


#Logistic Regression
clf_logistic = LogisticRegression(penalty="l2", solver='lbfgs')
clf_logistic.fit(x_train, y_train)


# # MLP

# In[220]:


#MLP Classifier
clf_nn = MLPClassifier(hidden_layer_sizes = (100, 100, 50, 50, 20))
clf_nn.fit(x_train, y_train)


# In[ ]:


#NN Train - without feature extraction
train_data = []
train_labels = []
for file in os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Data_New\\"):
        data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Data_New\\" + file)
        if (len(data.shape) > 1):
            data = data[:,0]
        train_data.append(data[0:10000])
        train_labels.append(0)
        
for file in os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Data_New\\"):
        data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Data_New\\" + file)
        if (len(data.shape) > 1):
            data = data[:,0]
        train_data.append(data[:10000])
        train_labels.append(1)


# In[ ]:


#NN Test - without feature extraction
test_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Eng_Sur.flac")
test_data = test_data[:,0]
test_data = test_data[:10000]
print(test_data.shape)
test_data = numpy.reshape(test_data, (1, 10000))
pred = clf_nn.predict_proba(test_data)
print(pred)


# # GMM

# In[201]:


#GMM
clf_gmm = GaussianMixture()
clf_gmm.fit(x_train, y_train)
# print(y_train)


# # RNN

# In[196]:


#RNN
rnn_model = Sequential()
print(len(x_train))
# Embedding layer
print(x_train.shape)
rnn_model.add(Embedding(input_dim = 26,
              input_length = len(x_train),
              output_dim = 100,
              trainable = False,
              mask_zero = True))

# Masking layer for pre-trained embeddings
rnn_model.add(Masking(mask_value=0.0))

# Recurrent layer
rnn_model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
rnn_model.add(Dense(64, activation='relu'))

# Dropout for regularization
rnn_model.add(Dropout(0.5))

# Output layer
rnn_model.add(Dense(6, activation='softmax'))

# Compile the model
rnn_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = rnn_model.fit(x_train, Y, epochs=150)


# # Testing

# In[223]:


print(clf_logistic.score(x_train, y_train))
print(clf_logistic.score(x_test, y_test))


# In[228]:


# print(clf_nn.predict(x_test))
clf_gmm = pk.load(open('./pickled\\clf_gmm_multi_class', 'rb'))
pred = clf_gmm.predict(x_test)
print(accuracy_score(pred, y_test))
pred = clf_gmm.predict(x_train)
print(accuracy_score(pred, y_train))

# print(clf_nn.score(x_test, y_test))
# pred = clf.predict(x_test)

# print(accuracy_score(pred, y_test))


# In[229]:


clf_svm = pk.load(open('./pickled\\clf_svm_multi_class', 'rb'))
print(clf_svm.score(x_train, y_train))
print(clf_svm.score(x_test, y_test))


# In[230]:


print(clf_nn.score(x_train, y_train))
print(clf_nn.score(x_test, y_test))


# In[ ]:


plot_roc_curve_multi_class(normalized_X, Y, 6, clf_svm)


# In[235]:





# In[98]:


testing('C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Hindi_Data_New\\Hindi3.flac', clf_svm)


# In[56]:


# mfcc_features = []
# delta_features = []
# test_Hindi = []
# data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Test\\Eng_Sur.flac")
# print(data.shape)
# print(samplerate)
# data = data[:,0]
# mfcc_features, delta_features = extract_features(data, samplerate)
# test_Hindi = numpy.concatenate((mfcc_features,delta_features),axis = 1)
# test_Hindi = preprocessing.normalize(test_Hindi)
# results = clf_svm.predict(test_Hindi)
# print(results)
# print(sum(results))
# print (sum(results)/len(test_Hindi))


# # TSNE

# In[27]:


tsne_images = TSNE(n_components=2).fit_transform(normalized_X)
tsne_classifier = MLPClassifier()
print(1)
tsne_classifier.fit(tsne_images, Y)
print(2)
plot_decision_regions(tsne_images, (numpy.asarray(Y)).astype(int), clf = tsne_classifier, legend = 2)


# In[ ]:


plot_support_vectors(tsne_images, (numpy.asarray(Y)).astype(int), tsne_classifier)


# In[52]:


def plot_support_vectors(X, Y, clf):
    
    cmap_type = plt.cm.Paired
    s_val = 30
    
    plt.scatter(X[:, 0], X[:, 1], s = s_val, c = Y, cmap = cmap_type)

    current_axes = plt.gca()
    xlim = current_axes.get_xlim()
    ylim = current_axes.get_ylim()
    xx = numpy.linspace(xlim[0], xlim[1], s_val)
    yy = numpy.linspace(ylim[0], ylim[1], s_val)
    YY, XX = numpy.meshgrid(yy, xx)
    xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 100,
               linewidth = 1, facecolors = 'none', edgecolors = 'k')

    #reference taken from: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html


# In[215]:


data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Eng_Abhi.flac")
data = data[:,0]
hin_mfcc, hin_delta = extract_features(data, samplerate)
data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Eng_sur.flac")
data = data[:,0]
eng_mfcc, eng_delta = extract_features(data, samplerate)

zeros = [0]*len(x_English)

# zeros = [0]*len(eng_mfcc)
eng_mfcc = numpy.array(eng_mfcc)
eng_delta = numpy.array(eng_delta)
# x_English = numpy.concatenate((eng_mfcc, eng_delta), axis = 1)
ones = [1]*len(hin_mfcc)
hin_mfcc = numpy.array(hin_mfcc)
hin_delta = numpy.array(hin_delta)
x_Hindi = numpy.concatenate((hin_mfcc, hin_delta), axis = 1)
# x_Hindi = hin_delta
X = numpy.concatenate((x_English, x_Hindi), axis=0)
Y = numpy.concatenate((zeros, ones), axis=0)
normalized_X = preprocessing.normalize(X)

tsne_images = TSNE(n_components=2).fit_transform(normalized_X)
tsne_classifier = svm.SVC()
tsne_classifier.fit(tsne_images, Y)
plot_decision_regions(tsne_images, (numpy.asarray(Y)).astype(int), clf = tsne_classifier, legend = 2)


# In[26]:


audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Hindi_Data_flac\\h2.flac")
print(samplerate)


# # Multi Class

# In[105]:


top_coder_data = pd.read_csv('C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Data\TopCoder_Data\\trainingData.csv')
top_coder_data = numpy.array(top_coder_data)
languages = top_coder_data[:,1]
files = top_coder_data[:,0]
indices = []
labels = []
language_array = ["Hindi", "Kannada", "Dutch", "Arabic", "Korean South", "Thai"]
count = 0
num = 20
for lang in language_array:
    indices.extend((list(numpy.where(languages == lang)[0]))[:num])
    labels.extend([count]*num)
    count+=1
samples = files[indices]


# In[104]:


print(labels)


# In[86]:


picklevar("list_multi_class", list(samples))
print(samples)


# In[106]:


print(labels)


# In[126]:


mfcc_features = []
delta_features = []
features = []
label = []
num_iterations = 0

for file in tqdm(samples):
    file = file[:-4] + ".flac"
    audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Multi_Class_flac\\" + file)
    print(file, samplerate)
    audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
    mfcc_features.extend(audio_mfcc)
    delta_features.extend(audio_delta)
    label.extend([labels[num_iterations]]*len(audio_mfcc))
    num_iterations+=1

mfcc_features = numpy.array(mfcc_features)
delta_features = numpy.array(delta_features)
# X = numpy.concatenate((mfcc_features, delta_features)
# Y = label
# normalized_X = preprocessing.normalize(X)


# In[133]:


mfcc_features = numpy.array(mfcc_features)
delta_features = numpy.array(delta_features)
print(mfcc_features.shape, delta_features.shape)
X = numpy.concatenate((mfcc_features, delta_features), axis=1)
Y = label
normalized_X = preprocessing.normalize(X)
# print(Y)
print(X.shape, len(Y))


# In[134]:


x_train, x_test, y_train, y_test = train_test_split(normalized_X, Y, test_size = 0.2)


# In[103]:


picklevar("normalized_X_multi_class", normalized_X)
picklevar("Y_multi_class", Y)


# In[92]:


for i in range(len(labels)):
    print(labels[i], samples[i])


# In[213]:


for file in os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\Hindi_Data_2\\"):
    audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Data_2\\" + file)
    print(file, samplerate)

