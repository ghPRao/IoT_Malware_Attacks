# %%
'''
    Run this notebook for creating all models
    
'''
%load_ext autoreload

import os
import sys
sys.path.append("../..")  #Path to data directory
import pandas as pd
import collections
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.impute import SimpleImputer

from sklearn.utils import resample
# from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Added below imports to disable some warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from src.model_prep import *
from src.data_constants import *

# %%
"""
# House Keeping
"""

# %%
# Python library yellowbrick is used for multi-class visualization
#pip install yellowbrick

# %%
"""
# Business Understanding
Botnet-powered distributed denial of service (DDoS) attacks have used infected IoT devices to bring down websites. IoT devices can also be used to direct other attacks. For instance, there may be future attempts to weaponize IoT devices. A possible example would be a nation shutting down home thermostats in an enemy state during a harsh winter.

In response, network intrusion detection systems have been developed to detect suspicious network activity. Routers are trained to identify cyber threats.

This project will predict  Machine Learning models to detect Botnet attacks on IoT devices

Datasets available in UCI.
https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT#
https://archive.ics.uci.edu/ml/machine-learning-databases/00442/

"""

# %%
"""
# Data Preparation
"""

# %%
"""
### Exploring observation data:
```
Combined observations from all traffic types and devices have 7574739 observations. Data preparation is done through device class object instatiation time when you call IoT_Device method in src/model_prep.py.

Data preparation is done at IoT_Device class object initiation time.
1) Loading 95 datasets ( 5 benign datasets and 10 malware datasets for each IoT device) in to a dataframe 
2) Create targeg variable 'Class' to identify benign, mirai and gafgyt data
3) All Mirai and gafyt malware data is combined in to their respective dataframe.
4) To avoid class imbalance, a new dataset is created with equal benign, mirai, and gafgyt data.
5) Combined dataframe is split to features and target data into X and y.
6) Data is stored in the class itself and pickled for future use to avoid redoint the above steps

For the Samsung Webcam, there was no Mirai attack data, so assume that Mirai didn't infect these cameras. Between the three type of traffics --benign, mirai, and bashlite-- you will notice a class imbalance. Although for cyber attack like these, typically you will expect a lot of threat records compared to normal traffic, however it depends on number of devices infected. 
```

<p align="center">
  <img width="800" height="800" src="../../visualization/images/data_explore.jpeg">
</p>

"""

# %%
"""
## Devices list
Dataset from UCI were collected from 9 IoT Devices in 3 categories listed below.
For each device, data is organized under type of attack (Benign, Mirai, Bashlite). 
Benign traffic is normal traffic which are not attacked by botnet malware.
Mirai and Bashlite are two types of malware attacking these iot devices.
For the putpose of this project, the botnet name Gyfgyt and Bashlite are used interchangeably. 
For each device the following data is collcted.
- 1) Benign data
- 2) Mirai Malware Threat
- 3) Gyfgyt Malware Threat
"""

# %%
# List of all IoT devices for which data was collcted
device_list = list_iot_devices()

# %%
"""
## Running Models 
> Because of large datasets, I will read all data files once in a class instance and **pickle** the dataset to avoid reading several times.

> For each device I will run number of models to predict malware attacks.

> As a first step, run models for one devic **Damini Doorbell**

> Following models are run for comparision
> 1. DummyClassifier (Typically favors majority of the class)
> 2. Logistic Regression - FSM (Use all data - no train/test split and use defaults)
> 3. Logistic Regression       (30% test data is held. Remaining 70% data is split to train and validation)
> 4. KNeighborsClassifie       (    ,,   )
> 5. RandomForestClassifier    ( ,, ) -> Computationally expensive, so not running it all the time.
> 6. XGBClassifier             (,, )

### Model Evaluation
> A malware attack is consequential to the network and cyber security threat to consumers and business alike. <BR>
> I will use **Recall** as the primary method to evaluate the model to reduce Type-II error. Also, overall **Accuracy** is considered
"""

# %%
"""
## For each device we will run number of models to predict malware attacks.
"""

# %%
"""
## 1) Device: Damini_Doorbell

 
> Instantiate class IoT_Device.<BR>
1) Call ** read_device_pickle** to deserialize iot class object if a pickle file exis and revive the class instance to iot<BR>
2) If a pickle file aready exist, prompt is given whether to create it again.<BR>
3) If no pickle file exisits, instantiate an object by calling **IoT_Device** and write it to a new **pickle** file
"""

# %%
# Read the pickle file or intantiate IoT_Device
damini_iot = read_device_pickle(DAMINI_DOORBELL)   

# If the file doesn't exist or request to recreat, instantiate and write the pickle file
if damini_iot == None:
    # Instantiate class IoT_Device
    damini_iot = IoT_Device(DAMINI_DOORBELL)
    
    #Pickle the object
    write_device_pickle(damini_iot, DAMINI_DOORBELL)

# %%
"""
### Model 0: Dummmy Model
> Dummy model will most likely favor the majority class
"""

# %%
damini_iot.iot_Model("DummyClassifier")

# %%
"""
> Model pretty much pickedup all the values in the respective classes as benign traffic <BR>
> Now lets run a LogisticRegression with pretty much
"""

# %%
"""
### Model 1: FSM LogisticRegression
> Now lets run a LogisticRegression model with pretty much all defaults with out split
"""

# %%
# Run plain vanila FSM LogisticRegression model with all default inpts - binary target
damini_iot.iot_Model("FSM")

# %%
"""
### First Simple accuracy is low, though it appears recall score is 0.
> Prediction on benign traffic is low<BR>
> Even much lower or no Mirai atacks were identified<BR>
> More traffic was identified as infected with Gafgyt malware, which is incorrect.
    
> The traffic measurements for each of those devices will need scaling before we pick-up the right signals from the data.
"""

# %%
"""
## Model2: LogisticRegression -- Target = binary; Benign(=0), Mirai(=1), Gafgyt(=2)

- 1) Train-Test-Split with 30% test data
- 2) Scale the predictor dataset 
- 3) Fit LogisticRegression model with mostly default values and run cross_validation_score, confustion_matrix for train and test data.
"""

# %%
# Run plain vanila FSM LogisticRegression model with all default inpts - binary target
damini_iot.iot_Model("LogisticRegression")

# %%
import collections
print("      Train: ",collections.Counter(damini_iot.lg_train_prediction))
print(" Validation: ",collections.Counter(damini_iot.lg_val_prediction))
print("       Test: ", collections.Counter(damini_iot.lg_test_prediction))
print("      Label: ",collections.Counter(damini_iot.y_test))

# %%
"""
> Looks very good. 2.73% False Negatives nad 0.05% False positives <BR>
> Gafgyt traffic's are identified more accurately than the Mirai attaks.<BR>
> Overall Accuracy of 0.972, Type 1 errors (false positives)0.05%. <BR>

"""

# %%
"""
## Model 3: KNeighborsClassifier -- Target = binary; Benign(=0), Mirai(=1), Gafgyt(=2)
> After running this, found it is ccomuputationally very expensive
> It's Accuracy is lot lower than Logistic Regeression.
"""

# %%
damini_iot.iot_Model("KNeighborsClassifier")

# %%
print("\n      Train: ",collections.Counter(damini_iot.knn_train_prediction))
print(" Validation: ",collections.Counter(damini_iot.knn_val_prediction))
print("       Test: ", collections.Counter(damini_iot.knn_test_prediction))
print("      Label: ",collections.Counter(damini_iot.y_test))
print("\n\n\n")

# %%
"""
### Model 4: "DecisionTreeClassifier" -- Target = binary; Benign(=0), Mirai(=1), Gafgyt(=2)
"""

# %%
damini_iot.iot_Model("DecisionTreeClassifier")

# %%
print("\n      Train: ",collections.Counter(damini_iot.dt_train_prediction))
print(" Validation: ",collections.Counter(damini_iot.dt_val_prediction))
print("       Test: ", collections.Counter(damini_iot.dt_test_prediction))
print("      Label: ",collections.Counter(damini_iot.y_test))
print("\n\n\n")

# %%
"""
## Model 4: RandomForestClassifier
> Run GridSearchcv with parameters { 'max_depth': [ 2, 5, 10 ], 'n_estimators': [ 100, 1000, 2000]}
"""

# %%
damini_iot.iot_Model("RandomForestClassifer")

# %%
print("\n      Train: ",collections.Counter(damini_iot.rfc_train_prediction))
print(" Validation: ",collections.Counter(damini_iot.rfc_val_prediction))
print("       Test: ", collections.Counter(damini_iot.rfc_test_prediction))
print("      Label: ",collections.Counter(damini_iot.y_test))
print("\n\n\n")

# %%
# damini_iot.iot_compare_recall_accuracy("KNeighborsClassifier")

# %%
"""
# Model 5: XGBoost
"""

# %%
# XGBoost
damini_iot.iot_Model("XGBClassifier")

# %%
print("\n      Train: ",collections.Counter(damini_iot.xgb_train_prediction))
print(" Validation: ",collections.Counter(damini_iot.xgb_val_prediction))
print("       Test: ", collections.Counter(damini_iot.xgb_test_prediction))
print("      Label: ",collections.Counter(damini_iot.y_test))
print("\n\n\n")

# %%
"""
# 1) * Run LogisticalRegression, KNN, DecisionTree, RandomForest, and XGBoos for all 9 IoT devices
"""

# %%
def model_factory(iot):
    '''
    Function: model_factory
    Input: class instantiated when IoT_Device class was instantiated - this correspond to requested IoT evice
    Action: Run all 5 models, print confusion matrix, print key statistics
    '''
    iot.iot_Model("LogisticRegression")
    print("      Train: ", collections.Counter(iot.lg_train_prediction))
    print(" Validation: ", collections.Counter(iot.lg_val_prediction))
    print("       Test: ", collections.Counter(iot.lg_test_prediction)) 
    print("      Label: ", collections.Counter(iot.lg_test_prediction)) 
    print("\n\n")

    iot.iot_Model("KNeighborsClassifier")
    print("      Train: ",collections.Counter(iot.knn_train_prediction))
    print(" Validation: ",collections.Counter(iot.knn_val_prediction))
    print("       Test: ", collections.Counter(iot.knn_test_prediction))  
    print("      Label: ", collections.Counter(iot.knn_test_prediction)) 
    print("\n\n")    
    
    iot.iot_Model("DecisionTreeClassifier")
    print("      Train: ",collections.Counter(iot.dt_train_prediction))
    print(" Validation: ",collections.Counter(iot.dt_val_prediction))
    print("       Test: ", collections.Counter(iot.dt_test_prediction))    
    print("      Label: ", collections.Counter(iot.dt_test_prediction)) 
    print("\n\n") 
    
    iot.iot_Model("RandomForestClassifer")
    print("      Train: ",collections.Counter(iot.rfc_train_prediction))
    print(" Validation: ",collections.Counter(iot.rfc_val_prediction))
    print("       Test: ", collections.Counter(iot.rfc_test_prediction))
    print("      Label: ",collections.Counter(iot.y_test))
    print("\n\n")
    
    iot.iot_Model("XGBClassifier")
    print("      Train: ",collections.Counter(iot.xgb_train_prediction))
    print(" Validation: ",collections.Counter(iot.xgb_val_prediction))
    print("       Test: ", collections.Counter(iot.xgb_test_prediction))
    print("      Label: ",collections.Counter(iot.y_test))
    print("\n\n")

# %%
"""
# 2) *** Device: Ennino Doorbell ***
> Instantiate class IoT_Device.

>> Read datafiles and reate predictor and target data stored in the class
>> Call IoT_Device methods run predit models
"""

# %%
# Read the pickle file or intantiate IoT_Device
ennino_iot = read_device_pickle(ENNINO_DOORBELL) 
print("Pickle file for Ennino Doorbell is already created.\nDo you want to overwrite it? [Yes/No]", )
if ennino_iot == None:
    # Instantiate class IoT_Device
    ennino_iot = IoT_Device(ENNINO_DOORBELL)
    
    #Pickle the object
    write_device_pickle(ennino_iot, ENNINO_DOORBELL)
model_factory(ennino_iot)

# %%
"""
# 3) * Ecobee_Thermostat *
"""

# %%
# Read the pickle file or intantiate IoT_Device
ecobee_iot = read_device_pickle(ECOBEE_THERMOSTAT)    
if ecobee_iot == None:
    # Instantiate class IoT_Device
    ecobee_iot = IoT_Device(ECOBEE_THERMOSTAT)
    
    #Pickle the object
    write_device_pickle(ecobee_iot, ECOBEE_THERMOSTAT)
model_factory(ecobee_iot)

# %%
"""
# 4) * Philips_B120N10_Baby_Monitor *
"""

# %%
# Read the pickle file or intantiate IoT_Device
B120N10_iot = read_device_pickle(PHILIPS_B120N10_BABYMONITOR)    
if B120N10_iot == None:
    # Instantiate class IoT_Device
    B120N10_iot = IoT_Device(PHILIPS_B120N10_BABYMONITOR)
    
    #Pickle the object
    write_device_pickle(B120N10_iot, PHILIPS_B120N10_BABYMONITOR)
model_factory(B120N10_iot)

# %%
"""
# 5) Provision_PT_737E_Security_Camera
"""

# %%
# Read the pickle file or intantiate IoT_Device
PT737E_iot = read_device_pickle(PROVISION_737E_SECURITY_CAMERA)    
if PT737E_iot == None:
    # Instantiate class IoT_Device
    PT737E_iot = IoT_Device(PROVISION_737E_SECURITY_CAMERA)
    
    #Pickle the object
    write_device_pickle(PT737E_iot, PROVISION_737E_SECURITY_CAMERA)
model_factory(PT737E_iot)

# %%
"""
# 6) Provision_PT_838_Security_Camera
"""

# %%
# Read the pickle file or intantiate IoT_Device
PT838_iot = read_device_pickle(PROVISION_838_SECURITY_CAMERA )    
if PT838_iot == None:
    # Instantiate class IoT_Device
    PT838_iot = IoT_Device(PROVISION_838_SECURITY_CAMERA )
    
    #Pickle the object
    write_device_pickle(PT838_iot, PROVISION_838_SECURITY_CAMERA )
model_factory(PT838_iot)

# %%
"""
SimpleHome_XCS7_1002_WHT_Security_Camera# 7) Samsung_SNH_1011_N_Webcam
"""

# %%
"""
# 7) Samsung SNH-1011N Webcam
"""

# %%
# Read the pickle file or intantiate IoT_Device
SAMSUNG_iot = read_device_pickle(SAMSUNG_1011N_WEBCAM )    
if SAMSUNG_iot == None:
    # Instantiate class IoT_Device
    SAMSUNG_iot = IoT_Device(SAMSUNG_1011N_WEBCAM )
    
    #Pickle the object
    write_device_pickle(SAMSUNG_iot, SAMSUNG_1011N_WEBCAM )
model_factory(SAMSUNG_iot)

# %%
"""
# 8) SimpleHome XCS7-1002WHT Security Camera
"""

# %%
# Read the pickle file or intantiate IoT_Device
XCS7_1002_iot = read_device_pickle(SIMPLEHOME_1002_SECURITY_CAMERA )    
if XCS7_1002_iot == None:
    # Instantiate class IoT_Device
    XCS7_1002_iot = IoT_Device(SIMPLEHOME_1002_SECURITY_CAMERA )
    
    #Pickle the object
    write_device_pickle(XCS7_1002_iot, SIMPLEHOME_1002_SECURITY_CAMERA )
model_factory(XCS7_1002_iot)

# %%
"""
# 9) SimpleHome XCS7-1003WHT Security Camera
"""

# %%
# Read the pickle file or intantiate IoT_Device
XCS7_1003_iot = read_device_pickle(SIMPLEHOME_1003_SECURITY_CAMERA )    
if XCS7_1003_iot == None:
    # Instantiate class IoT_Device
    XCS7_1003_iot = IoT_Device(SIMPLEHOME_1003_SECURITY_CAMERA )
    
    #Pickle the object
    write_device_pickle(XCS7_1003_iot, SIMPLEHOME_1003_SECURITY_CAMERA)
model_factory(XCS7_1003_iot)

# %%
"""
## Create a list of all IoT_Device instantiated class variables
> This list is used in a loop to iterate through all devices
"""

# %%
#list of all IoT_Devices intantiated class variable
iot_classObjs = [damini_iot, ennino_iot, ecobee_iot, B120N10_iot, SAMSUNG_iot, PT737E_iot, PT838_iot, XCS7_1002_iot, XCS7_1003_iot]

# %%
df = pd.DataFrame()
for item, device in enumerate(DEVICE_LST):
    iot = iot_classObjs[item]
    df = df.append(iot.lg_model_score, ignore_index=True)
    df = df.append(iot.knn_model_score, ignore_index=True)
    df = df.append(iot.dt_model_score, ignore_index=True)
    df =  df.append(iot.rfc_model_score, ignore_index=True)
    df = df.append(iot.xgb_model_score, ignore_index=True)

# print Recall_0
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Recall_0', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.99, 1.0))
ss.set(ylabel = 'Benign: Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Recall_0
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Recall_1', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.99, 1.0))
ss.set(ylabel = 'Mirai: Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Recall_2
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Recall_2', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.990, 1.0))
ss.set(ylabel = 'gafgyt: Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Macro Recall
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Macro_Recall', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.990, 1.0))
ss.set(ylabel = 'Macro Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Accuracy
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Accuracy', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.990, 1.0))
ss.set(ylabel = 'Accuracy')
ss.set(xticklabels=[])
ss.set(xlabel=None)


# %%
df = pd.DataFrame()
for item, device in enumerate(DEVICE_LST):
    iot = iot_classObjs[item]

    df =  df.append(iot.rfc_model_score, ignore_index=True)
    df = df.append(iot.xgb_model_score, ignore_index=True)

# print Recall_0
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Recall_0', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.99, 1.0))
ss.set(ylabel = 'Benign: Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Recall_0
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Recall_1', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.99, 1.0))
ss.set(ylabel = 'Mirai: Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Recall_2
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Recall_2', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.990, 1.0))
ss.set(ylabel = 'gafgyt: Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Macro Recall
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Macro_Recall', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.990, 1.0))
ss.set(ylabel = 'Macro Recall')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# print Accuracy
plt.figure(figsize=(18, 10))
ss = sns.barplot(x = 'Device', y = 'Accuracy', hue='Model', data = df, ci=None);
ss.set_box_aspect(5/len(ss.patches)) #change 10 to modify the y/x axis ratio
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ss.set(ylim=(0.990, 1.0))
ss.set(ylabel = 'Accuracy')
ss.set(xticklabels=[])
ss.set(xlabel=None)

# %%
print("Mean RFC Accuracy: ", np.mean(df[df['Model'] == "RFC"]['Accuracy']))

print("Mean RFC Macro REcall: ", np.mean(df[df['Model'] == "RFC"]['Macro_Recall']))

print("Mean XGB Accuracy: ", np.mean(df[df['Model'] == "XGB"]['Accuracy']))

print("Mean XGB Macro Recall: ", np.mean(df[df['Model'] == "XGB"]['Macro_Recall']))

# print( " Mean: Accuracy ", np.mean(df['Accuracy']))
# print( " Mean: Macro Recall", np.mean(df['Macro_Recall']))    


# %%
"""
# * Best Model Evaluation * XGBoost *

> For each of the IoT device, do typeper parameter turning
  > "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] <BR>
  > "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],<BR>
  > "min_child_weight" : [ 1, 3, 5, 7 ],<BR>
  > "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],<BR>
  > "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]<BR>

## Call iot_xgboost_hyperparameter_tuning()  method in data_prep.py
"""

# %%
'''
    Set XGBoost Hyperparameters and run RandomizedSearachCV to find the best parameters
'''
iot_classObjs = [damini_iot, ennino_iot, ecobee_iot, B120N10_iot, SAMSUNG_iot, PT737E_iot, PT838_iot, XCS7_1002_iot, XCS7_1003_iot]

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}

for iot in iot_classObjs:
    iot.iot_xgboost_hyperparameter_tuning('recall', params)
#    iot.iot_xgboost_hyperparameter_tuning('accuracy', params)

# %%
'''
Method Name: iot_xgboost_hyperparameter_tuning
    Input: self
    Returns: None
    Because I have a large dataset, let me put some timer while running parameter turning
    Loop through each of the 9 IoT devices to score on Recall.
'''
for iot in iot_classObjs:
    # Instantiate XGBClassifer()
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.5, gamma=0.4, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.25, max_delta_step=0, max_depth=8,
                  min_child_weight=1,  monotone_constraints='()',
                  n_estimators=100, n_jobs=0, num_parallel_tree=1,
                  objective='multi:softprob', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=None, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None, score="")
    # random_search = RandomizedSearchCV(xgb,param_distributions=params,n_iter=5,scoring='recall_macro',  n_jobs=-1,cv=5,verbose=3)

    score = cross_val_score(xgb, iot.X,iot.y,cv=10)
print("Score: ", score)

# if which_score == 'recall':
#     print("\XGBoost - Mean of Macro Recall scores:",score.mean())
# else:
#     print("\nXGBoost - Mean of Accuracy scores: ", score.mean())

# %%
"""
## Run randomforest_hypterparameter_tuning
"""

# %%
iot_classObjs = [damini_iot, ennino_iot, ecobee_iot, B120N10_iot, SAMSUNG_iot, PT737E_iot, PT838_iot, XCS7_1002_iot, XCS7_1003_iot]
params = {"n_estimators": [100,300,500,700, 900, 1000],
          "criterion": ["gini", "entropy"],
          "max_depth": [ 3,  5,  10, None], 
          "max_features": ["auto", "sqrt"] }
for iot in iot_classObjs:
    iot.iot_randomforest_hyperparameter_tuning('recall', params)
    iot.iot_randomforest_hyperparameter_tuning('accuracy', params)    

# %%
"""
# Final Model: RandomForestClassifier
    Using Best Parameters: {'min_child_weight': 1, 
                            'max_depth': 15, 
                            'learning_rate': 0.05, 
                            'gamma': 0.4, 
                            'colsample_bytree': 0.7)
"""

# %%
xgbRecall = [0.9997510841201962, 0.9995993179880648, 0.9993644886366209, 0.9998269040553908, 
            0.9996356663470756, 0.9997157590315533, 0.9997597640816961, 0.999656541292868, 0.99902703284789]

xgbAccuracy = [0.9997510841201962, 0.9995993179880648, 0.9993644886366209, 0.9998269040553908,
               0.9996356663470756, 0.9997157590315533, 0.9997597640816961, 0.999656541292868,
               0.99902703284789]


rfcRecall = [ 0.9997578126895196, 0.9995566922421142, 0.999517005159244, 0.9997926652971163,
             0.9996548418024929, 0.9996835814110652, 0.999665022367665, 0.999592143137277,
             0.9989758266572988]



print("XGB Hyper Tuning Recall Results: min-", min(xgbRecall), "  max-",max(xgbRecall), "mean-", np.mean(xgbRecall))
print("XGB Hyper Tuning Accuracy Results: min-", min(xgbAccuracy), "  max-",max(xgbAccuracy), "mean-", np.mean(xgbAccuracy))
print("RF Hyper Tuning Recall Results: min-", min(rfcRecall), "  max-",max(rfcRecall), "mean-", np.mean(rfcRecall))

# %%
