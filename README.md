
# 🌸Sakura Bloom Prediction with Neural Networks and Machine Learning🌸

###### A quantitative comparison of LSTM networks and simpler machine learning approaches for forecasting Kyoto's Cherry Blossom phenology.

## 🌸 Description

This project presents a systematic comparison between neural network architectures and a regression model for predicting the peak bloom date of cherry blossoms (Prunus × yedoensis) in Kyoto, Japan. Using 145 years (1881-2025) of daily meteorological records and phenological observations from the Japan Meteorological Agency and Osaka Prefecture University scholars. Four approaches were evaluated: Long Short-term Memory network,  multi-layer perceptron (MLP), XGBoost, and Ridge regression.

## 🌸 Project Importance

* Tourism: The hanami (花見, cherry blossom viewing) season drives Japan's spring tourism economy, attracting approximately 63 million visitors and generating an estimated $2.7 billion in revenue annually.
* Cultural significance: Sakura, or cherry blossoms, are a profound symbol in Japanese culture, representing the beauty and transience of life, renewal, and the arrival of spring. The cherry blossom is considered the national flower of Japan, and is central to the custom of hanami - a centuries-old tradition where people gather under the trees to contemplate and celebrate the blossom.
* Climatological value: The flowering time of cherry trees is thought to be affected by global warming and the heat island effect of urbanization, therefore bloom timing serves as a sensitive indicator of climate change.


## 🌸 The Problem Statement

### Objective

Predict the bloom date of cherry blossoms in Tokyo for specific test years using three different models and historical weather data from the Japanese Meteorological Agency,

### Dataset

The dataset was created using two data-sources: the Kyoto time-series with cherry blossom peak days of blooming was taken from Yasuyuki Aono’s research paper, phenological and meteorological data was acquired from the Japanese Meteorological Agency. The time period from 1881 to 2025 was selected, since the temperature data for earlier years is inconsistent. 
Based on the obtained minimum, maximum, and average temperature, other weather-related features were built. The day of the year with a peak blossom was obtained from the dates from Yasuyuki Aono’s data and chosen as the model's target to predict based on the dataset features. The preprocessing included changing data format, and feature engineering. Then, the scaling was applied to standardize the data. Additionally, the dataset was split into train(1881-11993), test(1994-2007), and test(2008-2025) subsets.

### Models to Compare

Three models were chosen for this project:
LSTM: theoretically, is considered to be the best architecture for time-series prediction, for its capacity to weight the value importance and update its memory. 
MLP: another model, which is considered to perform well on time-series forecasting. 
XGB: 
Ridge Regression: one of machine learning models chosen to compare the prediction capability and performance to the neural networks.

## 🌸 The Models Implemented

### LSTM

Recurrent neural networks designed to capture temporal dependencies in sequential data.

* Input: meteorological and phenological daily data from 1881 to 1998(test dataset)
* Output: a predicted peak bloom day of the year

LSTM Architecture:

#### Key Components and Hyperparameters:

1. Optimizer: Adam
2. Criterion: MSE Loss 
3. Hidden dimension: 32
4. Number of layers: 1	
5. Dropout: 0
6. Learning rate: 0.0001

### MLP

A feedforward neural network to capture non-linear patterns in the data.

* Input: meteorological and phenological daily data from 1881 to 1998(test dataset)
* Output: a predicted peak bloom day of the year
 
  
MLP Architecture:

    Input Layer → Hidden Layer 1 (32, ReLU) → Hidden Layer 2 (16, ReLU) → Output Layer (1, Linear)

#### Key Components and Hyperparameters:
* Optimizer: Adam
* Criterion: MSE Loss  
* Hidden dimension: 32
* Number of layers: 1	
* Dropout: 0
* Learning rate: 0.0001

### XGBoost

 A tree-based ensemble method included for comparison with neural network approaches.

* Input: Aggregated meteorological and phenological daily features (1881-1993 training, 1994-2007 test)
* Output: Predicted peak bloom day-of-year
### 
    ŷᵢ = Σₖ₌₁ᴷ fₖ(xᵢ), fₖ ∈ F, where ŷᵢ - predicted bloom day for year i, 
    K - number of trees, 
    fₖ - k-th decision tree, 
    F - space of regression trees, 
    xᵢ - input features for year i (16-dimensional vector)

Each tree fₖ maps features to a score:
### 
    fₖ(x) = w_{q(x)}, where q(x) - function mapping input to leaf node, 
    w - leaf weight (prediction value at that leaf)


### Ridge Regression

To address multicollinearity and enable use of daily temperatures.

* Input: meteorological and phenological daily data from 1881 to 1993(test dataset)
* Output: a predicted peak bloom day of the year

Ridge Regression is a linear model with L2 regularization that minimizes:
### 
    β̂ = argminₚ Σᵢ (yᵢ - Xᵢβ)² + λ Σⱼ βⱼ², where yᵢ - actual bloom day for year i, Xᵢ - feature vector for year i, 
    β - coefficient vector, λ - regularization strength, Σⱼ βⱼ² - L2 penalty 

Once trained, predictions are a simple linear combination:
### 
    ŷ = Xβ = β₀ + β₁x₁ + β₂x₂ + ... + βᵢxᵢ, where ŷ - predicted bloom day, β₀ - intercept term, β₁...βᵢ - feature weights,
    x₁...xᵢ = input features


## 🌸 Evaluation Metrics
### RMSE 
    √[(1/N) Σ (ŷ_i - y_i)²] in days, 
    where N - sample size, ŷ - the predicted value for an i-year, 
    yᵢ - the actual observed value for an i-year.

### MSE
    (1/N) Σ (ŷᵢ - yᵢ)², where N - sample size, ŷ - the predicted value for an i-year, 
    yᵢ - the actual observed value for an i-year.

### MAE
    (1/N) Σ |ŷ_i - y_i|, where N - sample size, ŷ - the predicted value for an i-year,
    yᵢ - the actual observed value for an i-year.

## 🌸 Results & Performance
#### Comparative Analysis (Test Set: 2001-2024)



| Model         | MSE           |  MAE         |  RMSE  |
| ------------- |:-------------:|:-------------:|:--:|
| LSTM          | 9.3334     |   2.692461       |  3.055  | 
| MLP           | 10.9559     |   10.9559      |  3.3099  |
| Ridge Regression       | 12.6186     |   12.618579      |  3.5523  |
|XGBoost|      4732.0210        |   61.6177    |  68.7897  |

### Key Performance Insights
#### LSTM
* The best performance out of all the models
* Fails the most to predict the date for warmer years
* Tends to generalize the learned data 
* Limited by temperature-based only features
* Unstable, requires a lot of data and a really cautious parameter configuration
* Potential improves with a wider range of data

#### MLP
* Shows strong performance
* Performs well on normal years but struggles with extreme events, as well as LSTM
* Simpler architecture with fewer parameters than LSTM, reducing overfitting risk
* Slightly worse than LSTM (3.31 vs 3.06 days) – the gap quantifies the value of sequential modeling
* Faster to train and easier to optimize than recurrent architectures

#### XGBoost

* Demonstrates catastrophic failure on this task
* Cannot extrapolate beyond the range of training data
* Treats each year independently, ignoring temporal dependencies
* Complex feature engineering cannot compensate for lack of sequential memory, it also could become the reason for such a poor model performance: XGBoost could perceive addittional features as noise.

#### Ridge Regression
* Shows itself well in capturing correlation and linear trends in data
* The simplest implementation 
* Slightly worse than neural networks
* Not able to capture complex data patterns
* Does not generalize as much as the deep learning methods

## 🌸 Summary of the Results
### Phenological Insights 
* The temperature trend consistently showed the average temperature increase aver the years.
* The data reflects a global warming trend.
* Throughout the years, blooming appeared earlier and earlier.
### Dataset 
* Temperature is Primary Driver.
* A small range of training samples is insufficient for neural networks.
* The utilization of a bigger dataset might improve the neural networks results.

### The Models' Performance
* LSTM achieves best overall performance 
* MLP performs competitively but lacks temporal modeling
* Regression models provide interpretable baselines but XGBoost completely fails to predict the actual dates or the dates close to the actual ones, which is probably caused by a great number of the features used. 
* Neural Networks tend to generalize, while linear regression models try to fit to find the colinear data patterns and fit to a predictive linear function.

## 🌸 References

1. Aono Y, Saito S. Clarifying springtime temperature reconstructions of the medieval period by gap-filling the cherry blossom phenological data series at Kyoto, Japan. Int J Biometeorol. 2010 Mar;54(2):211-9. doi: 10.1007/s00484-009-0272-x. Epub 2009 Oct 23. PMID: 19851790. 

2. Aono, Y. and Kazui, K. (2008), Phenological data series of cherry tree flowering in Kyoto, Japan, and its application to reconstruction of springtime temperatures since the 9th century. Int. J. Climatol., 28: 905-914. https://doi.org/10.1002/joc.1594

3. Chmielewski FM, Götz KP. ABA and Not Chilling Reduces Heat Requirement to Force Cherry Blossom after Endodormancy Release. Plants (Basel). 2022 Aug 4;11(15):2044. doi: 10.3390/plants11152044. PMID: 35956522; PMCID: PMC9370221.

4. E., Jung & Y., Kwon & Chung, Uran & I., Yun. (2005). Predicting Cherry Flowering Date Using a Plant Phonology Model. Korean Journal of Agricultural and Forest Meteorology. 7. 

5. Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long Short-Term Memory. Neural Computation. 9. 1735-1780. 10.1162/neco.1997.9.8.1735. 

6. Miyawaki-Kuwakado, A., Han, Q., Kitamura, K., & Satake, A. (2024). Impacts of climate change on the transcriptional dynamics and timing of bud dormancy release in Yoshino-cherry tree. Plants, People, Planet, 6(6), 1505–1521. https://doi.org/10.1002/ppp3.10548

The JMA weather data: https://www.data.jma.go.jp/risk/obsdl/index.php

The dataset of peak bloom dates in Kyoto: https://www.kaggle.com/datasets/willianoliveiragibin/japans-cherry







     