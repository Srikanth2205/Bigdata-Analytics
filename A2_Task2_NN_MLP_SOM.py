#!/usr/bin/env python
# coding: utf-8

# In[268]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[269]:


data = pd.read_csv("A1_BC_SEER_data.csv")


# In[270]:


data.head()


# In[271]:


data.describe()


# In[272]:


data.info()


# In[273]:


data["survived"] = np.where(data["Survival months"]>=60, 1, 0)
data.head()


# In[274]:


new_data = data.copy()


# In[275]:


#Dropping non essential and specified columns
new_data.drop(["No."], axis=1, inplace=True)
new_data.drop(["Patient ID"], axis=1, inplace=True)
new_data.drop(["Survival months"], axis=1, inplace=True)


# In[276]:


new_data.head()


# In[277]:


#recoding categorical variables
cat_columns = {"SEER registry" : {1: "San Francisco-Oakland SMSA (1975+)", 
                                  2: "Connecticut (1975+)", 
                                  20 : "Metropolitan Detroit (1975+)", 
                                  21 : "Hawaii (1975+)", 
                                  22 : "Iowa (1975+)", 
                                  23 : "New Mexico (1975+)", 
                                  25 : "Seattle (Puget Sound) (1975+)", 
                                  26 : "Utah (1975+)", 
                                  27 : "Metropolitan Atlanta (1975+)", 
                                  29 : "Alaska (1992+)", 
                                  31 : "San Jose-Monterey (1992+)", 
                                  35 : "Los Angeles (1992+)", 
                                  37 : "Rural Georgia (1992+)", 
                                  41 : "Greater California (excl. SF, Los Ang. & SJ) (2000+)", 
                                  42 : "Kentucky (2000+)", 
                                  43 : "Louisiana (2000+)", 
                                  44 : "New Jersey (2000+)", 
                                  47 : "Greater Georgia (excluding AT and RG)", 
                                  61 : "Idaho* (2000+)", 
                                  62 : "New York* (2000+)", 
                                  63 : "Massachusetts* (2000+)"
                                 },
               
                "Sex" : {1 : "Male", 
                         2 : "Female", 
                         9 : "Sex Not stated (unknown)"
                        }, 
               "Marital status at diagnosis" : {1 : "Single (never married)", 
                                                2 : "Married (including common law)",
                                                3 : "Separated", 
                                                4 : "Divorced", 
                                                5 : "Widowed", 
                                                6 : "Unmarried or domestic partner (same sex or opposite sex or unregistered)", 
                                                9 : "Marital Status Unknown", 
                                                14 : "Marital Status Blank"
                                               }, 
               "Race/ethnicity" : {1 : "White", 
                                   2 : "Black", 
                                   3 : "American Indian, Aleutian, Alaskan Native or Eskimo (includes all indigenous populations of the Western hemisphere)", 
                                   4 : "Chinese", 
                                   5 : "Japanese", 
                                   6 : "Filipino", 
                                   7 : "Hawaiian", 
                                   8 : "Korean (Effective with 1/1/1988 dx)", 
                                   10 : "Vietnamese (Effective with 1/1/1988 dx)", 
                                   11 : "Laotian (Effective with 1/1/1988 dx)", 
                                   12 : "Hmong (Effective with 1/1/1988 dx)", 
                                   13 : "Kampuchean (including Khmer and Cambodian) (Effective with 1/1/1988 dx)", 
                                   14 : "Thai (Effective with 1/1/1994 dx)", 
                                   15 : "Asian Indian or Pakistani, NOS (Effective with 1/1/1988 dx)", 
                                   16 : "Asian Indian (Effective with 1/1/2010 dx)", 
                                   17 : "Pakistani (Effective with 1/1/2010 dx)", 
                                   20 : "Micronesian, NOS (Effective with 1/1/1991)", 
                                   21 : "Chamorran (Effective with 1/1/1991 dx)", 
                                   22 : "Guamanian, NOS (Effective with 1/1/1991 dx)", 
                                   25 : "Polynesian, NOS (Effective with 1/1/1991 dx)", 
                                   26 : "Tahitian (Effective with 1/1/1991 dx)", 
                                   27 : "Samoan (Effective with 1/1/1991 dx)", 
                                   28 : "Tongan (Effective with 1/1/1991 dx)", 
                                   30 : "Melanesian, NOS (Effective with 1/1/1991 dx)",
                                   31 : "Fiji Islander (Effective with 1/1/1991 dx)", 
                                   32 : "New Guinean (Effective with 1/1/1991 dx)", 
                                   96 : "Other Asian, including Asian, NOS and Oriental, NOS (Effective with 1/1/1991 dx)", 
                                   97 : "Pacific Islander, NOS (Effective with 1/1/1991 dx)", 
                                   98 : "Race/Ethnicity Other", 
                                   99 : "Race/Ethnicity Unknown"
                                  }, 
               "Histology recode - broad groupings" : {0 : "8000-8009 : unspecified neoplasms",
                                                       1 : "8010-8049 : epithelial neoplasms, NOS", 
                                                       2 : "8050-8089 : squamous cell neoplasms", 
                                                       3 : "8090-8119 : basal cell neoplasms", 
                                                       4 : "8120-8139 : transitional cell papillomas and carcinomas", 
                                                       5 : "8140-8389 : adenomas and adenocarcinomas", 
                                                       6 : "8390-8429 : adnexal and skin appendage neoplasms", 
                                                       7 : "8430-8439 : mucoepidermoid neoplasms", 
                                                       8 : "8440-8499 : cystic, mucinous and serous neoplasms", 
                                                       9 : "8500-8549 : ductal and lobular neoplasms", 
                                                       10 : "8550-8559 : acinar cell neoplasms", 
                                                       11 : "8560-8579 : complex epithelial neoplasms", 
                                                       12 : "8580-8589 : thymic epithelial neoplasms", 
                                                       13 : "8590-8679 : specialized gonadal neoplasms", 
                                                       14 : "8680-8719 : paragangliomas and glumus tumors", 
                                                       15 : "8720-8799 : nevi and melanomas", 
                                                       16 : "8800-8809 : soft tissue tumors and sarcomas, NOS",
                                                       17 : "8810-8839 : fibromatous neoplasms", 
                                                       18 : "8840-8849 : myxomatous neoplasms", 
                                                       19 : "8850-8889 : lipomatous neoplasms", 
                                                       20 : "8890-8929 : myomatous neoplasms", 
                                                       21 : "8930-8999 : complex mixed and stromal neoplasms", 
                                                       22 : "9000-9039 : fibroepithelial neoplasms", 
                                                       23 : "9040-9049 : synovial-like neoplasms", 
                                                       24 : "9050-9059 : mesothelial neoplasms", 
                                                       25 : "9060-9099 : germ cell neoplasms", 
                                                       26 : "9100-9109 : trophoblastic neoplasms", 
                                                       27 : "9110-9119 : mesonephromas", 
                                                       28 : "9120-9169 : blood vessel tumors", 
                                                       29 : "9170-9179 : lymphatic vessel tumors", 
                                                       30 : "9180-9249 : osseous and chondromatous neoplasms", 
                                                       31 : "9250-9259 : giant cell tumors", 
                                                       32 : "9260-9269 : miscellaneous bone tumors (C40._,C41._)", 
                                                       33 : "9270-9349 : odontogenic tumors ( C41._)", 
                                                       34 : "9350-9379 : miscellaneous tumors", 
                                                       35 : "9380-9489 : gliomas", 
                                                       36 : "9490-9529 : neuroepitheliomatous neoplasms", 
                                                       37 : "9530-9539: meningiomas", 
                                                       38 : "9540-9579 : nerve sheath tumors", 
                                                       39 : "9580-9589 : granular cell tumors & alveolar soft part sarcomas", 
                                                       40 : "9590-9599 : malignant lymphomas, NOS or diffuse", 
                                                       41 : "9650-9669 : hodgkin lymphomas", 
                                                       42 : "9670-9699 : nhl - mature b-cell lymphomas", 
                                                       43 : "9700-9719 : nhl - mature t and nk-cell lymphomas", 
                                                       44 : "9720-9729 : nhl - precursor cell lymphoblastic lymphoma", 
                                                       45 : "9730-9739 : plasma cell tumors", 
                                                       46 : "9740-9749 : mast cell tumors", 
                                                       47 : "9750-9759 : neoplasms of histiocytes and accessory lymphoid cells", 
                                                       48 : "9760-9769 : immunoproliferative diseases", 
                                                       49 : "9800-9805: leukemias, nos", 
                                                       50 : "9820-9839 : lymphoid leukemias (C42.1)", 
                                                       51 : "9840-9939 : myeloid leukemias (C42.1)", 
                                                       52 : "9940-9949 : other leukemias (C42.1)", 
                                                       53 : "9950-9969 : chronic myeloproliferative disorders (C42.1)", 
                                                       54 : "9970-9979 : other hematologic disorders", 
                                                       55 : "9980-9989 : myelodysplastic syndrome", 
                                                       98 : "HISTOLOGY RECODE Other"
                                                      }, 
               "Primary Site" : {500 : "Nipple", 
                                 501 : "Central portion of breast", 
                                 502 : "Upper-inner quadrant of breast", 
                                 503 : "Lower-inner quadrant of breast", 
                                 504 : "Upper-outer quadrant of breast", 
                                 505 : "Lower-inner quadrant of breast", 
                                 506 : "Auxillary tail of breast",
                                 508 : "Overlapping lesion of breast", 
                                 509 : "Breast NOS"
                                 }, 
               "Laterality" : {0 : "Not a paired site", 
                               1 : "Right: origin of primary", 
                               2 : "Left: origin of primary", 
                               3 : "Only one side involved right or left origin unspecified", 
                               4 : "Bilateral involvement", 
                               5 : "Paired site midline tumor", 
                               9 : "Paired site but no information concerning laterality midline tumor"
                               }, 
               "Breast - Adjusted AJCC 6th Stage (1988-2015)" : {0 : "0", 
                                                                 1 : "0a", 
                                                                 2 : "0is", 
                                                                 10 : "I", 
                                                                 11 : "INOS", 
                                                                 12 : "IA", 
                                                                 13 : "IA1", 
                                                                 14 : "IA2", 
                                                                 15 : "IB", 
                                                                 16 : "IB1", 
                                                                 17 : "IB2", 
                                                                 18 : "IC", 
                                                                 19 : "IS", 
                                                                 20 : "IEA", 
                                                                 21 : "IEB", 
                                                                 22 : "IE", 
                                                                 23 : "ISA", 
                                                                 24 : "ISB", 
                                                                 30 : "II", 
                                                                 31 : "IINOS", 
                                                                 32 : "IIA", 
                                                                 33 : "IIB", 
                                                                 34 : "IIC", 
                                                                 35 : "IIEA", 
                                                                 36 : "IIEB", 
                                                                 37 : "IIE", 
                                                                 38 : "IISA", 
                                                                 39 : "IISB", 
                                                                 40 : "IIS", 
                                                                 41 : "IIESA", 
                                                                 43 : "IIES", 
                                                                 50 : "III", 
                                                                 51 : "IIINOS", 
                                                                 52 : "IIIA", 
                                                                 53 : "IIIB", 
                                                                 54 : "IIIC", 
                                                                 55 : "IIIEA", 
                                                                 56 : "IIIEB", 
                                                                 57 : "IIIE", 
                                                                 58 : "IIISA", 
                                                                 59 : "IIISB", 
                                                                 60 : "IIIS", 
                                                                 61 : "IIIESA", 
                                                                 62 : "IIIESB", 
                                                                 63 : "IIIES", 
                                                                 70 : "IV", 
                                                                 71 : "IVNOS", 
                                                                 72 : "IVA", 
                                                                 73 : "IVB", 
                                                                 74 : "IVC", 
                                                                 88 : "N/A", 
                                                                 90 : "OCCULT", 
                                                                 99 : "UNK Stage", 
                                                                 126 : "Blank"
                                                                 }
              }


# In[278]:


new_data = new_data.replace(cat_columns)
new_data.head()


# In[279]:


#Changing surg combine to categprical
new_data["surg combine"] = new_data["surg combine"].astype('object', copy = False)


# In[280]:


new_data["Year of birth"] = new_data["Year of diagnosis"] - new_data["Age at diagnosis"]


# In[281]:


new_data.loc[new_data['Year of diagnosis'] >= 200, 'Year of diagnosis'] = 2000 + (new_data['Year of diagnosis'] % 100)
new_data.loc[new_data['Year of diagnosis'] < 200, 'Year of diagnosis'] = 1900 + (new_data['Year of diagnosis'] % 100)


# In[282]:


#recoding the columns to order
new_data.loc[new_data['ER Status Recode Breast Cancer (1990+)'] == 2, 'ER Status Recode Breast Cancer (1990+)'] = -1
new_data.loc[new_data['ER Status Recode Breast Cancer (1990+)'] == 3, 'ER Status Recode Breast Cancer (1990+)'] = 0
new_data.loc[new_data['ER Status Recode Breast Cancer (1990+)'] == 4, 'ER Status Recode Breast Cancer (1990+)'] = 2
new_data.loc[new_data['ER Status Recode Breast Cancer (1990+)'] == 9, 'ER Status Recode Breast Cancer (1990+)'] = 3


# In[283]:


new_data.loc[new_data['PR Status Recode Breast Cancer (1990+)'] == 2, 'PR Status Recode Breast Cancer (1990+)'] = -1
new_data.loc[new_data['PR Status Recode Breast Cancer (1990+)'] == 3, 'PR Status Recode Breast Cancer (1990+)'] = 0
new_data.loc[new_data['PR Status Recode Breast Cancer (1990+)'] == 4, 'PR Status Recode Breast Cancer (1990+)'] = 2
new_data.loc[new_data['PR Status Recode Breast Cancer (1990+)'] == 9, 'PR Status Recode Breast Cancer (1990+)'] = 3


# In[284]:


#dropping since SEER registry and other year columns are non essential and repetitive(year columns due to age at diagnosis)
new_data.drop(["SEER registry"], axis=1, inplace=True)
new_data.drop(["Year of birth"], axis=1, inplace=True)
new_data.drop(["Year of diagnosis"], axis=1, inplace=True)


# In[285]:


new_data.dtypes


# In[286]:


new_data["Primary Site"].value_counts()


# In[287]:


new_data["Laterality"].value_counts()


# In[288]:


new_data["Breast - Adjusted AJCC 6th Stage (1988-2015)"].value_counts()


# In[289]:


new_data["ER Status Recode Breast Cancer (1990+)"].value_counts()


# In[290]:


new_data["PR Status Recode Breast Cancer (1990+)"].value_counts()


# In[291]:


new_data["surg combine"].value_counts()


# In[292]:


new_data.describe()


# In[293]:


new_data.hist(bins=50, figsize=(20,15))
plt.show()


# In[294]:


from pandas.plotting import scatter_matrix

attributes = ["survived", "Sequence number", "PR Status Recode Breast Cancer (1990+)", 
              "ER Status Recode Breast Cancer (1990+)", "Age at diagnosis"]
scatter_matrix(new_data[attributes], figsize=(12, 8))


# In[295]:


corr_matrix = new_data.corr()


# In[296]:


corr_matrix["survived"].sort_values(ascending=False)


# In[297]:


corr_matrix


# In[298]:


sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, annot=True)


# In[299]:


new_data.drop(["PR Status Recode Breast Cancer (1990+)"], axis=1, inplace=True)


# In[300]:


new_data["Age at diagnosis"].value_counts()
new_data.hist(column='Age at diagnosis')


# In[301]:


target = new_data["survived"]
new_data = new_data.drop("survived", axis = 1)


# In[302]:


one_hot_cols = ["Marital status at diagnosis", "Race/ethnicity", "Sex", 
                "Primary Site", "Laterality", "Histology recode - broad groupings", 
                "Breast - Adjusted AJCC 6th Stage (1988-2015)", "surg combine"
               ]

num_cols = ["Sequence number", "Record number recode", "ER Status Recode Breast Cancer (1990+)", "Age at diagnosis"]


# In[303]:


for item in one_hot_cols:
    one_hot = pd.get_dummies(new_data[item])  #one-hot encoding
    new_data = new_data.drop(item, axis = 1)
    new_data = new_data.join(one_hot)


# In[304]:


new_data.head()


# In[305]:


final_columns = new_data.columns


# In[306]:


from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler()
new_data = scaling.fit_transform(new_data)


# In[307]:


new_data.shape


# In[308]:


myseed=11   #Seed for the random number generator


# In[309]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Split the rest into a training and validation set
X, x_test, Y, y_test = train_test_split(new_data, target, test_size=.2, random_state=myseed)


# In[310]:


#Split the rest into a training and validation set
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=.125, random_state=myseed)


# In[311]:


x_train.shape


# In[312]:


x_val.shape


# # Neural Net Building

# In[313]:


import tensorflow as tf
from tensorflow import keras

tf.__version__


# In[314]:


from sklearn.model_selection import GridSearchCV

# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, momentum=0):
    #create model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(108, input_dim=108, activation='relu'))
    model.add(keras.layers.Dense(54, activation="relu"))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    optimizer = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
np.random.seed(myseed)


# In[47]:


# create model and GridSearch CV for Hyperparameter tuning
model =  keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
epochs = [10, 20, 50]
learn_rate = [0.001, 0.01, 0.1]
momentum = [0.4, 0.6, 0.9]
param_grid = dict(epochs=epochs, learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)


# In[48]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[49]:


model = keras.models.Sequential()
model.add(keras.layers.Dense(108, input_dim=108, activation='relu'))
model.add(keras.layers.Dense(54, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))


# In[50]:


model.summary()


# In[51]:


optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.4)

model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])


# In[52]:


history = model.fit(x_train, y_train, 
                    epochs=20,
                    validation_data=(x_val, y_val))


# In[53]:


import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


# # # TESTING Phase

# In[54]:


x_test.shape


# In[55]:


model.evaluate(x_test, y_test)


# In[56]:


y_proba = model.predict(x_test)
y_proba.round(2)


# In[58]:


y_pred = model.predict_classes(x_test)
y_pred


# # Model Diagnositics

# In[59]:


from sklearn.metrics import confusion_matrix,accuracy_score

con_mat = confusion_matrix(y_test, y_pred)
classifier_accuracy = accuracy_score(y_test, y_pred)


# In[267]:


con_mat


# In[61]:


classifier_accuracy


# In[69]:


#[row, column]
TP = con_mat[1, 1]
TN = con_mat[0, 0]
FP = con_mat[0, 1]
FN = con_mat[1, 0]


# In[62]:


from sklearn.metrics import precision_score, recall_score

precision_score(y_test, y_pred)


# In[63]:


recall_score(y_test, y_pred)


# In[64]:


from sklearn.metrics import f1_score

f1_score(y_test, y_pred)


# In[70]:


#Calculating Accuracy
accuracy = (TP + TN)/(TP + FP + TN + FN)
print(accuracy)


# In[71]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)


# In[265]:


def plot_PR_curve(pr, rc, label=None):
    plt.plot(rc, pr, linewidth=2, label=label)
    #plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

plot_PR_curve(precisions, recalls)
plt.show()


# In[73]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[74]:


from sklearn.metrics import roc_curve

FPR, TPR, THs = roc_curve(y_test, y_proba)


# In[75]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(FPR, TPR)
plt.show()


# In[76]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_proba)


# In[77]:


def plot_FPR_TPR_vs_threshold(FPR, TPR, thresholds):
    plt.plot(thresholds, FPR, "b--", label="False Positive rate")
    plt.plot(thresholds, TPR, "g-", label="True positive rate")
    plt.xlabel("Threshold")
    plt.legend(loc="center right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
plot_FPR_TPR_vs_threshold(FPR, TPR, THs)
plt.show()


# In[266]:


ax= plt.subplot()
sns.heatmap(con_mat, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Predicted : 0', 'Predicted : 1'])
ax.yaxis.set_ticklabels(['True : 0', 'True : 1'])


# In[79]:


# Missclassified samples


# In[114]:


missclassified = []
wrong_predictions = []
for row_index, (input, prediction, label) in enumerate(zip (x_test, y_pred, y_test)):
    if prediction != label:
        #print('Row', row_index, 'has been classified as ', prediction, 'and should be ', label)
        missclassified.append(row_index)
        wrong_predictions.append(prediction)


# In[110]:


mis_clf_set = np.asarray([x_test[i, ] for i in missclassified])


# In[133]:


wrong_pred = pd.Series(wrong_predictions)


# # Diagnostics Using SOM

# In[89]:


from myminisom import MiniSom   #see Moodle site for myminisom
from copy import copy

#Create the SOM
som_shape = (60, 60)      #define the size of the som
som_test = MiniSom(som_shape[0], som_shape[1], x_test.shape[1], sigma=som_shape[0]/2, learning_rate=.9, neighborhood_function='gaussian', random_seed=myseed)

#initialize the SOM, then train it
epochs=20
som_test.pca_weights_init(x_test)
som_test.train_random(x_test, epochs * len(x_test), verbose=True)

# In[117]:


#Find the BMU for each sample
BMU_test = np.array([som_test.winner(x) for x in x_test])
BMU_class0 = BMU_test[y_test==0]
BMU_class1 = BMU_test[y_test==1]


# In[250]:


BMU_class1.shape


# In[137]:


#Find the BMU for each missclassified sample
BMU_miss = np.array([som_test.winner(x) for x in mis_clf_set])
BMU_miss0 = BMU_miss[wrong_pred==0]
BMU_miss1 = BMU_miss[wrong_pred==1]


# In[249]:


BMU_miss1.shape


# In[139]:


som_shape = (60, 60)      #define the size of the som
suffix = "+proportion"


# In[252]:


densitymap = np.zeros(som_shape)
for row in range(0,BMU_test.shape[0]):
    x,y = BMU_test[row]
    densitymap[y,x] += 1

densitymap[densitymap==0]=np.nan  #mask zero values
my_cmap = copy(plt.cm.jet)
my_cmap.set_bad(color=(1,1,1))
plt.figure(dpi=300)
plt.imshow(densitymap, cmap=my_cmap, interpolation="none", origin="lower", aspect=0.75)
plt.colorbar()
plt.title('Mapping density' + suffix)
plt.savefig('Mapping density ' +  suffix + '.png', dpi=300)
plt.show()


# In[253]:


densitymap = np.zeros(som_shape)
for row in range(0,BMU_miss.shape[0]):
    x,y = BMU_miss[row]
    densitymap[y,x] += 1

densitymap[densitymap==0]=np.nan  #mask zero values
my_cmap = copy(plt.cm.jet)
my_cmap.set_bad(color=(1,1,1))
plt.figure(dpi=300)
plt.imshow(densitymap, cmap=my_cmap, interpolation="none", origin="lower", aspect=0.75)
plt.colorbar()
plt.title('Mapping density' + suffix +' for missclassified samples')
plt.savefig('Mapping density ' +  suffix + ' for missclassified samples.png', dpi=300)
plt.show()


# In[254]:


#density map of all samples from class 1
densitymap = np.zeros(som_shape)
for row in range(0,BMU_class1.shape[0]):
    x,y = BMU_class1[row]
    densitymap[y,x] += 1

densitymap[densitymap==0]=np.nan  #mask zero values
plt.figure(dpi=300)
plt.imshow(densitymap, cmap=my_cmap, interpolation="none", origin="lower", aspect=0.75)
plt.colorbar()
plt.title('Mapping density (class 1)' + suffix)
plt.savefig('Mapping density class 1' +  suffix + '.png', dpi=300)
plt.show()


# In[255]:


#density map of all samples missclassified as class 0
densitymap = np.zeros(som_shape)
for row in range(0,BMU_miss0.shape[0]):
    x,y = BMU_class0[row]
    densitymap[y,x] += 1

densitymap[densitymap==0]=np.nan  #mask zero values
plt.figure(dpi=300)
plt.imshow(densitymap, cmap=my_cmap, interpolation="none", origin="lower", aspect=0.75)
plt.colorbar()
plt.title('Mapping density (Missclassified as class 0)' + suffix)
plt.savefig('Mapping density Missclassified as class 0' +  suffix + '.png', dpi=300)
plt.show()


# In[256]:


#density map of all samples from class 0
densitymap = np.zeros(som_shape)
for row in range(0,BMU_class0.shape[0]):
    x,y = BMU_class0[row]
    densitymap[y,x] += 1

densitymap[densitymap==0]=np.nan  #mask zero values
plt.figure(dpi=300)
plt.imshow(densitymap, cmap=my_cmap, interpolation="none", origin="lower", aspect=0.75)
plt.colorbar()
plt.title('Mapping density (class 0)' + suffix)
plt.savefig('Mapping density class 0' +  suffix + '.png', dpi=300)
plt.show()


# In[257]:


#density map of all samples missclassified as 1
densitymap = np.zeros(som_shape)
for row in range(0,BMU_miss1.shape[0]):
    x,y = BMU_class1[row]
    densitymap[y,x] += 1

densitymap[densitymap==0]=np.nan  #mask zero values
plt.figure(dpi=300)
plt.imshow(densitymap, cmap=my_cmap, interpolation="none", origin="lower", aspect=0.75)
plt.colorbar()
plt.title('Mapping density (Missclassified as class 1)' + suffix)
plt.savefig('Mapping density Missclassified as class 1' +  suffix + '.png', dpi=300)
plt.show()


# In[260]:


cols = list(final_columns)


# In[263]:


#ambiguity at neuron 55, 55
weight = som_test.get_weights()[55,55]
weight = weight.reshape((1, 108))

insight = scaling.inverse_transform(weight)
values = (np.round(insight)).reshape(108,).tolist()

df = {'Columns' : cols, 'Values' : values}
df = pd.DataFrame(df)

pd.set_option('display.max_rows', df.shape[0]+1)
print(df)


# In[264]:


#ambiguity at neuron 45, 40
weight = som_test.get_weights()[45,40]
weight = weight.reshape((1, 108))

insight = scaling.inverse_transform(weight)
values = (np.round(insight)).reshape(108,).tolist()

df = {'Columns' : cols, 'Values' : values}
df = pd.DataFrame(df)

pd.set_option('display.max_rows', df.shape[0]+1)
print(df)


# In[ ]:




