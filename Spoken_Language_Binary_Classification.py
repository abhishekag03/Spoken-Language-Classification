#!/usr/bin/env python
# coding: utf-8

# In[211]:


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
import math
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
from sklearn.preprocessing import label_binarize
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn


# # Pickle files

# In[2]:


def picklevar(filename, varname):
    file = open('./pickled\\' + filename, 'wb')
    pk.dump(varname, file)
    file.close()


# # Extract Audio Features

# In[107]:


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

# In[143]:


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

# In[ ]:


def plot_roc_curve(x_test, y_true, classifier):
    fpr = dict()
    tpr = dict()
    y_score = classifier.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    area_under_curve = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve area = %0.2f' %area_under_curve)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.show()
    


# # Audio Segmentation

# In[89]:


english_mixed_features=[]
hindi_mixed_features=[]
mixed_mfcc_features=[]
mixed_delta_features=[]
mixed_labels=[]
english=os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Mixed_Data\\")
hindi=os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Mixed_Data\\")
count=len(english)
for i in range(2*count):
    if(i%2==0):
        audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Mixed_Data\\" + english[i//2])
        audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
        mixed_mfcc_features.extend(audio_mfcc)
        mixed_delta_features.extend(audio_delta)
        list_English=[0]*len(audio_mfcc)
        mixed_labels.extend(list_English)
    else:
        audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Mixed_Data\\" + hindi[(i-1)//2])
        audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
        mixed_mfcc_features.extend(audio_mfcc)
        mixed_delta_features.extend(audio_delta)
        list_Hindi=[1]*len(audio_mfcc)
        mixed_labels.extend(list_Hindi)


# In[90]:


mixed_mfcc_features = numpy.array(mixed_mfcc_features)
mixed_delta_features = numpy.array(mixed_delta_features)
print(mixed_mfcc_features.shape,mixed_delta_features.shape)
X_mixed = numpy.concatenate((mixed_mfcc_features,mixed_delta_features), axis=1)
Y_mixed = mixed_labels
normalized_X_mixed = preprocessing.normalize(X_mixed)
print(normalized_X_mixed.shape, len(Y_mixed))


# # Get language transition times and calculate loss

# In[94]:


predict_output=list(clf_svm.predict(normalized_X_mixed))
true_output=Y_mixed
pred_acc=[]
true_acc=[]
i=0
while(i<len(predict_output)):
    pred_out=predict_output[i:i+65]
    true_out=true_output[i:i+65]
    sum_pred=sum(pred_out)/len(pred_out)
    sum_true=sum(true_out)/len(true_out)
    pred_acc.append(sum_pred)
    true_acc.append(sum_true)
    i=i+65
print(pred_acc,true_acc)
answers=[]
starts=[]
ends=[]
ans=0
glob = 0
for i in range(1,len(pred_acc)):
    if(pred_acc[i-1]<0.5 and pred_acc[i]>0.5):
        if((i+1)!=len(pred_acc) and pred_acc[i+1]>0.5):
            answers.append(i)
            starts.append(0)
            ends.append(1)
        
    elif(pred_acc[i-1]>0.5 and pred_acc[i]<0.5):
        if((i+1)!=len(pred_acc) and pred_acc[i+1]<0.5):
            answers.append(i)
            starts.append(1)
            ends.append(0)
        
dictlang={0:"English",1:"Hindi"}
    
for i in range(len(answers)):
    print("Predicted Answer")
    print("Transition From: ",dictlang[starts[i]],dictlang[ends[i]])
    if (glob == 0):
        print("Time: ",15*65*answers[i]/1000,15*65*(answers[i]+1)/1000)
        glob+=1
    else:
        print("Time: ",15*65*answers[i]*9/25000,15*65*9*(answers[i]+1)/25000)
    print()

answers2=[]
starts=[]
ends=[]
ans=0
for i in range(1,len(true_acc)):
    if(true_acc[i-1]<0.5 and true_acc[i]>0.5):
        if((i+1)!=len(true_acc) and true_acc[i+1]>0.5):
            answers2.append(i)
            starts.append(0)
            ends.append(1)
            
    elif(true_acc[i-1]>0.5 and true_acc[i]<0.5):
        if((i+1)!=len(true_acc) and true_acc[i+1]<0.5):
            answers2.append(i)
            starts.append(1)
            ends.append(0)

for i in range(len(answers2)):
    print("True Answer")
    print("Transition From: ",dictlang[starts[i]],dictlang[ends[i]])
    if (glob == 1):
        print("Time: ",15*65*answers[i]/1000,15*65*(answers[i]+1)/1000)
        glob+=1
    else:
        print("Time: ",15*65*answers[i]*9/25000,15*65*9*(answers[i]+1)/25000)
#     print("Time: ",15*65*answers2[i]/1000,15*65*(answers2[i]+1)/1000)
    print()

rmse=0
for i in range(len(answers)):
    
    rmse+=(answers2[i]-answers[i])**2

    
rmse=math.sqrt(rmse/len(answers))

print("Root Mean Square Error: ", rmse)


# # 2 Class langauge Detection

# In[21]:


english_mfcc_features = []
english_delta_features = []
english_features = []
num_iterations = 0
for file in tqdm(os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Mixed_Data_Train\\")):
    audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\English_Mixed_Data_Train\\" + file)
    audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
    english_mfcc_features.extend(audio_mfcc)
    english_delta_features.extend(audio_delta)
    num_iterations+=1


# In[22]:


hindi_mfcc_features = []
hindi_delta_features = []
hindi_features = []
num_iterations = 0
for file in tqdm(os.listdir("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Mixed_Data_Train\\")):
    audio_data, samplerate = sf.read("C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\Hindi_Mixed_Data_Train\\" + file)
    hamming_window = numpy.hamming(400)
    audio_mfcc, audio_delta = extract_features(audio_data, samplerate)
    hindi_mfcc_features.extend(audio_mfcc)
    hindi_delta_features.extend(audio_delta)
    num_iterations+=1


# In[23]:


print(len(hindi_mfcc_features), len(hindi_delta_features))
print(len(english_mfcc_features), len(english_delta_features))
if (len(hindi_mfcc_features) != len(hindi_delta_features)):
    print("PROBLEM")


# # Form Feature vector

# In[24]:


zeros = [0]*len(english_mfcc_features)
english_mfcc_features = numpy.array(english_mfcc_features)
english_delta_features = numpy.array(english_delta_features)
x_English = numpy.concatenate((english_mfcc_features, english_delta_features), axis = 1)

ones = [1]*len(hindi_mfcc_features)
hindi_mfcc_features = numpy.array(hindi_mfcc_features)
hindi_delta_features = numpy.array(hindi_delta_features)
x_Hindi = numpy.concatenate((hindi_mfcc_features, hindi_delta_features),axis = 1)

X = numpy.concatenate((x_English, x_Hindi), axis=0)
Y = numpy.concatenate((zeros, ones), axis=0)
normalized_X = preprocessing.normalize(X)


# # Split Train and Test set

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(normalized_X, Y, test_size = 0.2)


# # Classifiers

# # Feed Forward Neural Network Keras

# In[48]:


X=x_train
Y=y_train
model = Sequential()
model.add(Dense(100, input_dim=26, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# Y = to_categorical(Y)
model.fit(X, Y, epochs=20, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


# # Test Accuracy

# In[54]:


scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(x_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(accuracy_score(rounded, y_test))


# # Multi Class

# In[ ]:


# 

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
output_class = numpy.argmax(model.predict(x_test),axis=1)r
true_class = numpy.argmax(Y_test,axis=1)
sum=0
for i in range(len(output_class)):
    if(output_class[i]==true_class[i]):
        sum+=1
print("Accuracy ", sum/len(output_class))
#print(numpy.argmax(model.predict(X),axis=1),numpy.argmax(Y,axis=1))


# In[109]:


print(list(Y))


# ## SVM

# In[96]:


#SVM
# parameters = [{'kernel': ['rbf'], 'gamma' : [0.001,0.05,0.1,10], 'C' : [0.001, 0.01, 0.1, 1,2.5,100]}]
# clf = GridSearchCV(svm.SVC(), parameters, cv = 5)
# clf = GridSearchCV(svm.SVC())
clf_svm = svm.SVC(probability = True)
clf_svm.fit(x_train, y_train)


# ## Logistic

# In[98]:


#Logistic Regression
clf_logistic = LogisticRegression(penalty="l2", solver='lbfgs')
clf_logistic.fit(x_train, y_train)


# ## MLP

# In[ ]:


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


# # KNN

# In[ ]:


clf_knn = KNeighborsClassifier(n_neighbors=7)
clf_knn.fit(x_train, y_train)


# # GMM

# In[ ]:


#GMM
clf_gmm = GaussianMixture(probability = True)
clf_gmm.fit(x_train, y_train)
# print(y_train)


# # ROC Curve - 2 Classes 

# In[105]:


plot_roc_curve(x_test, y_test, clf_svm)


# In[ ]:


plot_roc_curve(x_test, y_test, clf_nn)


# In[104]:


plot_roc_curve(x_test, y_test, clf_logistic)


# In[ ]:


plot_roc_curve(x_test, y_test, clf_gmm)


# In[121]:


l = x_train.reshape(x_train.shape[0],x_train.shape[1],1)


# In[122]:


l.shape


# In[136]:


lol = y_train.reshape(y_train.shape[0],1)


# In[141]:


x_train.shape


# In[210]:


x_train_new, y_train_new = create_dataset(x_train[0:1000:], 0)


# In[157]:


def create_dataset(data, y):
    X_train = []
    y_train = []
    for i in range(10, data.shape[0]):
        X_train.append(data[i-10:i])
        y_train.append(y)
    X_train, y_train = numpy.array(X_train), numpy.array(y_train)
    X_train = numpy.reshape(X_train, (X_train.shape[0], 10, 26))
    return X_train, y_train


# # RNN 

# In[202]:


x_train_new = numpy.reshape(x_train_new, (x_train_new.shape[0], 10, 26))


# In[ ]:





# In[209]:


print(x_train_new.shape)


# In[204]:


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (10, 26)))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 2))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the RNN to the Training set
regressor.fit(x_train_new, y_train[:990], epochs = 100, batch_size = 300)


# In[147]:


#RNN
rnn_model = Sequential()
print(len(x_train))
# Embedding layer
print(x_train.shape)
# rnn_model.add(Embedding(input_dim = (26,1),
#                     input_length = 1,
#               output_dim = 100,
#               trainable = True,
#               mask_zero = True))

# Masking layer for pre-trained embeddings
# rnn_model.add(Masking(mask_value=0.0))
# Recurrent layer
rnn_model.add(LSTM(64, batch_input_shape=(1, 26, 1) ,return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
rnn_model.add(Dense(64, activation='relu'))

# Dropout for regularization
# rnn_model.add(Dropout(0.5))

# Output layer
rnn_model.add(Dense(1, activation='softmax'))
rnn_model.summary()
# Compile the model
rnn_model.compile(
    optimizer='adam',loss = 'mse', metrics=['accuracy'])
history = rnn_model.fit(l, lol, epochs=150)


# # Testing

# In[27]:


print(clf_svm.score(x_test, y_test))
print(clf_svm.score(x_train, y_train))


# In[ ]:


print(clf_logistic.score(x_test, y_test))
print(clf_logistic.score(x_train, y_train))


# In[42]:


pred = clf_gmm.predict(x_test)
print(accuracy_score(pred, y_test))
pred = clf_gmm.predict(x_train)
print(accuracy_score(pred, y_train))


# In[56]:


print(clf_nn.score(x_test, y_test))
print(clf_nn.score(x_train, y_train))


# In[ ]:


print(clf_knn.score(x_test, y_test))
print(clf_knn.score(x_train, y_train))


# In[28]:


testing('C:\\Users\Lenovo\Desktop\\Utility\Semester 5\Machine Learning\Project\\English_Mixed_Data\\174-168635-0022.flac', clf_svm)


# # Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_test, pred)
mat = numpy.array(cm)
classes = [1,2,3,4,5,6]
confusion_matrix1(mat, classes)


# # TSNE Plots

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
# X = numpy.concatenate((mfcc_features, delta_features))
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


# In[92]:


for i in range(len(labels)):
    print(labels[i], samples[i])

