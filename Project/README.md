# **Signal mining based on machine learning**

## Team Member

Student Name | Student ID
:---------  | ---------
Li panyu| 2001212358
Li Linxiong| 2001212357


## Project Introduction

In the past 10 years, deep learning models based on neural networks have led the development of artificial intelligence. Different from traditional machine learning, deep learning models directly extract features from raw data and make predictions for targets in an `end-to-end` manner, thereby avoiding manual intervention and information loss in multi-step learning. However, when deep learning is applied to `multi-factor stock selection`, the effect of applying existing models may not meet expectations, and a suitable network structure needs to be tailored.    
In order to integrate the factor generation and multi-factor synthesis steps in multi-factor stock selection, this project designs two types of network structure to `predict the rise or fall` of each stock over the next 10 days inspired by Huatai Research report (see [here](https://github.com/PanyuLi/PHBS_MLF_2021/edit/main/Project/docs/Research_report_from_Huatai.pdf)): the first is `AlphaNet` using `raw price-volume data` as input, the other is using `price-volume data after feature extraction`, including: CNN, LogisticRegression and Random Forest.

## Data Analysis

### 1. Data Source
* Stock pool: CSI 300 component stocks from 08/10/2010 to 05/11/2021
* Data: volume and price data of individual stocks without feature engineering, transfer the volume and price data into 9*30 `data pictures `, and 30 is the number of historical days
* Labels: rise or fall of each stock over the next 10 days

**Raw data sample**  
<div align='center'>
    <img src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/datasample1.png' width='650'/>
</div>  

**Data sample after feature extraction**   
<div align='center'>
    <img src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/datasample2.png' width='700'/>
    </div>
 
### 2. Data processing

* Deal with missing data: open, high, low, close fill with the previous data; volume, pct_chg, vwap, turn, free_turn_n use 0 to fill NAN. After processing, the data has no NAN values. 
* One-hot encoding: One-hot encoder is used to convert categorical data into integer data.
* Feature Standardization：to avoid unit effect, we normalized the data along the feature direction  

**Raw data after dropna**   
<div align='center'>
    <img src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/dropna.png' width='600'/>
</div> 

### 3. Feature Correlation and Distributions

To get a better understanding of the properties of different features, we plot the features `correlations heatmap` and the `distributions` of the features. Through the corrletion graph, we can see that features from the same type has high correlation. Through the distributions graph, we can see that most of the features do not have a normal distribution.

**Feature correlation**   
<div align='center'>
    <img src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/heatmap.png' width='500'/>
</div>    

**Feature distribution**   
<div align='center'>
    <img src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/distribution.png' width='700'/>
    </div>
    
## Model
We devide our data set into training set and validate set. According to datetime, the former 80% belongs to the training set and the rest belongs to validate set. We devide data according to datetime in order to avoid the influence of so called future information.

### 1. Alphanet: use raw price-volume data as input
In order to effectively extract features from the original stock volume and price data, AlphaNet uses the idea of feature construction in genetic programming  and uses a variety of operator functions as a custom network layer for feature extraction. AlphaNet consists of four parts: 
* `Data input`: adopting CNN data format, the original volume and price data of individual stocks are sorted into two-dimensional "data pictures"
* `Feature extraction layer`: the most critical part of AlphaNet, it implemented a variety of custom computing network layers to extract features, and use the Batch Normalization layer for feature standardization.
* `Pooling layer`: consistent with pooling layer in CNN, the characteristics of the upper layer are "blurred".
* `Fully connected layer`: weighted synthesis of extracted features and uses `sigmoid` activation function to make classification

**Network structure**
<div align='center'><img width='700'src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/alphanet_model.png'/></div>

**Example show the process of feature extraction**  
<div align='center'><img width='600'src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/feature_extract.png'/></div>

**Network structure details**  

Layer name | Components | Parameters
:--- | :--- | :---
Feature extraction | ts_corr(X, Y, stride), ts_cov(X, Y, stride), ts_stddev(X, stride), ts_zscore(X, stride), ts_return(X, stride), ts_decaylinear(X, stride), ts_mean(X, stride), BN | stride=10
Pooling layer | ts_mean(X, stride), ts_max(X, stride), ts_min(X, stride), BN | stride=3
Fully Connected Layer | 30 neurons | dropout rate: 0.5, activation function: relu
Output Layer | 1 neurons | activation function: sigmoid
Other params | `Loss function`: Binary_cross_entropy, `Optimizer`: Adam, `Learning rate`: 0.001, `batch_size`: 256

**Toolkit: PyTorch**  
PyTorch is an open source machine learning library and is widely used in deep learning scenarios for its flexibility. PyTorch uses dynamic computational graphs rather than static graphs, which can be regarded as the mean difference between it and other deep learning frameworks.

**Confusion matrix**

<div align='center'><img width='300'src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/con_matrix_alphanet.png'/></div>

### 2. Other models: use the price-volume data after feature extraction as input

#### 2.1 Convolutional Neural Network

Convolutional Neural Network (CNN) is the most influential model in the field of computer vision research and application. The structure of Convolutional Neural Network mimics the working principle of the visual nerve of the eye. The optic nerve is the bridge of the it and the brain, the visual nerve of a lot of collaboration, each responsible for a small parts of the visual image, then the image of different combination of the local feature abstraction to the top of visual concept, make the human has the visual recognition ability Convolution neural network is also similar, includes input layer, hidden layer and output layer. The hidden layer is composed of `convolutional layer, pooling layer and full connection layer`. Feature extraction is carried out by multiple convolution kernels in the local area of the image, and dimension reduction is carried out by pooling layer, and finally feature synthesis is carried out by full connection layer.  
In this project, we build a `CNN with one convolutional layer, one max pooling layer with kernal (3, 3) and two fully connected layer with neurons 32, 1`, respectively.

**Network structure**  
<div align='center'><img width='400' src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/cnn_model.png'/></div>  

**Confusion matrix**  
<div align='center'><img width='300' src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/con_matrix_cnn.png'/></div>

#### 2.2 LogisticRegression
In order to reduce the time of computing, we use pca method to deduct dimensions. We observed that the 3 most import features can explain almost 90% of all the variance, so we used pca method and set components equal to 3.  
To improve the performance of models, we use GridSearch to find the best C value from [0.01, 0.1, 1, 10, 100].

**Confusion matrix**
<div align='center'><img width='300' src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/con_matrix_lr.png'/></div>

#### 2.3 RandomForest
To improve the performance of models, we use GridSearch to find the best max depth from [10, 20, 30, 40] and best max features from [5, 15, 25, 35, 45, 55].

**Confusion matrix**
<div align='center'><img width='300' src='https://github.com/PanyuLi/PHBS_MLF_2021/blob/main/Project/image/con_matrix_randomforest.png'/></div>


## Factor Validity Test

### 1.Rank_IC

Firstly, we calculate Rank IC of factors that we get from four different machine learning methods: Alphanet, CNN, Linear Logistic and Random Forest). Rank IC shows the relation between the rank of factors and the rank of asset return. **A high positive or a low negative correlation mean our factor is well explained the asset return and have a good prediction.**

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142371916-c795f69d-d8a6-46f9-95cc-228e28f04df0.png'/>
</div>

In addition, we check the standarlized IC which is called information ratio (IR):

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142371942-61bdd513-c190-47cf-a544-37e4f9e5bb9a.png'/>
</div>

In order to analyze the incremental information of the synthetic factor, we also show the test results after the market value is neutralized.  

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142371978-ccddef53-1c98-41cc-862f-05a346a33376.png'/>
</div>

**Rank IC and IR results**

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372494-5ee70526-b9e6-4c18-b53b-dc60215077dc.png'/>
</div>

**There must something wrong in alphanet since it has no correlation with return. After checking the factor data, we find CNN find nothing after learning. All factors of each stock are totally same.** 

We guess that there are several reasons for this failure：

- We do not use enough sample data, only 500-trading-day data, to train the module.

- Our computer does not have enough computing power to support the operation of this algorithm. 


**prediction result of Alphanet**

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372507-36795ce6-e1fa-4d16-9cef-72babdd905b2.png'/>
</div>

Therefore, in subsequent inspections and backtests, we will no longer test the Alphanet method 

**Cumsum of Rank_IC of each method**
<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372517-9727692a-c97d-400f-833c-34279cbe77f6.png'/>
</div>

**The synthetic factors do not have a pretty well explanation of asset returns. Compared with logistice regression and random forest, CNN has better correlation with asset returns with a highest mean of Rank_IC.**

### 2.Hierarchical backtest (market_value weighted)

We also conducted a hierarchical back-testing test. **The stocks are divided into five groups according to the order of factors from small to large**, and the market value of each group is weighted to analyze the net value trend during the test period. If the five groups of trends are stratified obviously, it means that the factor explains the asset return well. 

#### 2.1 CNN  Hierarchical backtest

The fifth group which the factors value are highest makes a much better performance than other four groups. However, The other four groups are not distinguished obviously.

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372537-934caebb-cf1e-4b6e-9eeb-ad9b5c1d0512.png'/>
</div>

#### 2.2 Logistic Regression Hierarchical backtest

The five groups devided by LR do not perform differently.
<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372545-f35e3e2e-aa8a-4bb0-be83-d8c479de107c.png'/>
</div>

#### 2.3 Random Forest Hierarchical backtest

The five groups devided by RF do not perform differently.
<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372559-a91f50f7-e399-45ef-ade1-f69c842552ee.png'/>
</div>

## Net value backtest

In this part, we make backtest of each maching leaning method and draw the net value curve to compare it with CSI 300 Index(IF300). The blue curve is the backetest result of our machine learning mehtod and the red curve is the cumulative net value of CSI 300 Index.

### 1. CNN net value backtest

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372571-74cf018e-7e0f-48cc-8253-07cc9f447664.png'/>
</div>

### 2. Logistic regression net value backtest

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372577-cec82bc2-3c67-46c5-9f4a-b5886e06d622.png'/>
</div>

### 3. Random forest net value backtest

Ramdom forset cannnot beat IF300 well.

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372584-4af9e04f-35d1-4990-9925-1cf926e276ad.png'/>
</div>

We analyze four important indexes which are annualized rate of return, annualized volatility, sharp ratio and maximum drawdown.

According to those indexed, we can conclude that CNN ,logistic regression and random forest do not perform so well since their sharp ratio are all around or less than 1.

<div align='center'>
  <img width='500'src='https://user-images.githubusercontent.com/61198794/142372593-37a34542-5889-45cd-b358-a52f31e292f2.png'/>
</div>
