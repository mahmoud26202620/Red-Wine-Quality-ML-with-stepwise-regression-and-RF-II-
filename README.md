# Red-Wine-Quality-ML-with-stepwise-regression-and-RF-II-

# Introduction

This case study is using Wine Quality Data Set from the UCI machine learning repository, in which the two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
Our research will focus on the red wine's quality, and it will be divided into two parts.

**1** -will attempt to predict whether the wine is good or not.

**2** -will try to estimate the precise quality of the wine as it appears in the dataset.

every part can be read by it's own, and this is the part two of the study. [Part one](https://github.com/mahmoud26202620/Red-Wine-Quality-ML-with-logistic-regression-and-RF-I-)

![Red_Wine_Glass](https://user-images.githubusercontent.com/41892582/227959684-1d4e9efc-56ff-440d-87a2-6483851273cc.jpg)

**Loading libraries**

Firstly I will start by loading some packages that I will use during the analysis

~~~
library(tidyverse)
library(Hmisc)
library(rms)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggcorrplot)
library(corrplot)
library(ppcor)
library(EFAtools)
library(splines)
~~~

**Getting the data**

~~~
wine<-read.csv("winequality-red.csv")
~~~

**Exploration of the data**

~~~
##the structure of the data
str(wine)
~~~

~~~
'data.frame':	1599 obs. of  12 variables:
 $ fixed.acidity       : num  7.4 7.8 7.8 11.2 7.4 7.4 7.9 7.3 7.8 7.5 ...
 $ volatile.acidity    : num  0.7 0.88 0.76 0.28 0.7 0.66 0.6 0.65 0.58 0.5 ...
 $ citric.acid         : num  0 0 0.04 0.56 0 0 0.06 0 0.02 0.36 ...
 $ residual.sugar      : num  1.9 2.6 2.3 1.9 1.9 1.8 1.6 1.2 2 6.1 ...
 $ chlorides           : num  0.076 0.098 0.092 0.075 0.076 0.075 0.069 0.065 0.073 0.071 ...
 $ free.sulfur.dioxide : num  11 25 15 17 11 13 15 15 9 17 ...
 $ total.sulfur.dioxide: num  34 67 54 60 34 40 59 21 18 102 ...
 $ density             : num  0.998 0.997 0.997 0.998 0.998 ...
 $ pH                  : num  3.51 3.2 3.26 3.16 3.51 3.51 3.3 3.39 3.36 3.35 ...
 $ sulphates           : num  0.56 0.68 0.65 0.58 0.56 0.56 0.46 0.47 0.57 0.8 ...
 $ alcohol             : num  9.4 9.8 9.8 9.8 9.4 9.4 9.4 10 9.5 10.5 ...
 $ quality             : int  5 5 5 6 5 5 5 7 7 5 ...
~~~

basic description of the data


~~~
##basic description of the data
describe(wine)
~~~

~~~
wine 

 12  Variables      1599  Observations
------------------------------------------------------------------------------------------
fixed.acidity 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       96    0.999     8.32    1.893      6.1      6.5      7.1      7.9 
     .75      .90      .95 
     9.2     10.7     11.8 

lowest :  4.6  4.7  4.9  5.0  5.1, highest: 14.3 15.0 15.5 15.6 15.9
------------------------------------------------------------------------------------------
volatile.acidity 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0      143        1   0.5278    0.199    0.270    0.310    0.390    0.520 
     .75      .90      .95 
   0.640    0.745    0.840 

lowest : 0.120 0.160 0.180 0.190 0.200, highest: 1.180 1.185 1.240 1.330 1.580
------------------------------------------------------------------------------------------
citric.acid 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       80    0.999    0.271   0.2227    0.000    0.010    0.090    0.260 
     .75      .90      .95 
   0.420    0.522    0.600 

lowest : 0.00 0.01 0.02 0.03 0.04, highest: 0.75 0.76 0.78 0.79 1.00
------------------------------------------------------------------------------------------
residual.sugar 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       91    0.996    2.539    1.078     1.59     1.70     1.90     2.20 
     .75      .90      .95 
    2.60     3.60     5.10 

lowest :  0.9  1.2  1.3  1.4  1.5, highest: 13.4 13.8 13.9 15.4 15.5
------------------------------------------------------------------------------------------
chlorides 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0      153        1  0.08747  0.03217   0.0540   0.0600   0.0700   0.0790 
     .75      .90      .95 
  0.0900   0.1090   0.1261 

lowest : 0.012 0.034 0.038 0.039 0.041, highest: 0.422 0.464 0.467 0.610 0.611
------------------------------------------------------------------------------------------
free.sulfur.dioxide 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       60    0.998    15.87    11.24        4        5        7       14 
     .75      .90      .95 
      21       31       35 

lowest :  1  2  3  4  5, highest: 55 57 66 68 72
------------------------------------------------------------------------------------------
total.sulfur.dioxide 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0      144        1    46.47    34.63     11.0     14.0     22.0     38.0 
     .75      .90      .95 
    62.0     93.2    112.1 

lowest :   6   7   8   9  10, highest: 155 160 165 278 289
------------------------------------------------------------------------------------------
density 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0      436        1   0.9967 0.002081   0.9936   0.9946   0.9956   0.9968 
     .75      .90      .95 
  0.9978   0.9991   1.0000 

lowest : 0.99007 0.99020 0.99064 0.99080 0.99084, highest: 1.00260 1.00289 1.00315 1.00320 1.00369
------------------------------------------------------------------------------------------
pH 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       89        1    3.311   0.1716     3.06     3.12     3.21     3.31 
     .75      .90      .95 
    3.40     3.51     3.57 

lowest : 2.74 2.86 2.87 2.88 2.89, highest: 3.75 3.78 3.85 3.90 4.01
------------------------------------------------------------------------------------------
sulphates 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       96    0.999   0.6581   0.1679     0.47     0.50     0.55     0.62 
     .75      .90      .95 
    0.73     0.85     0.93 

lowest : 0.33 0.37 0.39 0.40 0.42, highest: 1.61 1.62 1.95 1.98 2.00
------------------------------------------------------------------------------------------
alcohol 
       n  missing distinct     Info     Mean      Gmd      .05      .10      .25      .50 
    1599        0       65    0.998    10.42    1.178      9.2      9.3      9.5     10.2 
     .75      .90      .95 
    11.1     12.0     12.5 

lowest :  8.40000  8.50000  8.70000  8.80000  9.00000
highest: 13.50000 13.56667 13.60000 14.00000 14.90000
------------------------------------------------------------------------------------------
quality 
       n  missing distinct     Info     Mean      Gmd 
    1599        0        6    0.857    5.636   0.8431 

lowest : 3 4 5 6 7, highest: 4 5 6 7 8
                                              
Value          3     4     5     6     7     8
Frequency     10    53   681   638   199    18
Proportion 0.006 0.033 0.426 0.399 0.124 0.011
------------------------------------------------------------------------------------------
~~~

**Checking for NAs**

~~~
colSums(is.na(wine))
~~~
~~~
       fixed.acidity     volatile.acidity          citric.acid 
                   0                    0                    0 
      residual.sugar            chlorides  free.sulfur.dioxide 
                   0                    0                    0 
total.sulfur.dioxide              density                   pH 
                   0                    0                    0 
           sulphates              alcohol              quality 
                   0                    0                    0 
~~~

data is clean and ready for analys

We have to first divide the data into training and test sets before performing any analysis.

~~~
##make it reproducible
set.seed(113)
#use 80% of dataset as training set and 20% as test set
sample <- sample(c(TRUE, FALSE), nrow(wine), replace=TRUE, prob=c(0.8,0.2))
train.wine <- wine[sample, ]
test.wine <- wine[!sample, ]
~~~

**A First Look at the Data**

Distribution of wine quality

~~~
ggplot(wine,aes(x=quality,fill=as.factor(quality)))+
  geom_bar(width = 0.7)+
  ggtitle("Distribution of wine quality")+
  theme(legend.position = "none")
~~~

![Distribution of wine quality](https://user-images.githubusercontent.com/41892582/228034470-0e785e72-557e-402d-adca-8b409e288e37.jpg)

Evidently, the majority of wines are rated as being between 6 and 7, and only a small percentage are of bad ones or outstanding grade.


**using variables to investigate trends**

the relationship between quality and fixed acidity
~~~
ggplot(wine,aes(y=quality,x=fixed.acidity))+
  geom_smooth(method="loess")
~~~

![the relationship between quality and fixed acidity](https://user-images.githubusercontent.com/41892582/228036597-cbe6dd58-fe7a-4a79-8b86-30aeb32617cb.jpg)

the relationship between quality and volatile acidity

~~~
ggplot(wine,aes(y=quality,x=volatile.acidity))+
  geom_smooth(method="loess")
~~~
![2](https://user-images.githubusercontent.com/41892582/228037419-4f92b912-2d33-47b1-98b9-ab0aef906320.jpg)

the relationship between quality and citric acid
~~~
ggplot(wine,aes(y=quality,x=citric.acid))+
  geom_smooth(method="loess")
~~~
![3](https://user-images.githubusercontent.com/41892582/228041060-be1a39fd-b840-462a-b1d4-9e116b272f79.jpg)

the relationship between quality and residual.sugar
~~~
ggplot(wine,aes(y=quality,x=residual.sugar))+
  geom_smooth(method="loess")
~~~
![4](https://user-images.githubusercontent.com/41892582/228040931-891ac9c3-24cf-4bda-a3c9-cf61a5994f78.jpg)

the relationship between quality and chlorides
~~~
ggplot(wine,aes(y=quality,x=chlorides))+
  geom_smooth(method="loess")
~~~
![5](https://user-images.githubusercontent.com/41892582/228040985-99904db0-f34f-4ba2-9bf4-f48a91723f6c.jpg)

the relationship between quality and free sulfur dioxide
~~~
ggplot(wine,aes(y=quality,x=free.sulfur.dioxide))+
  geom_smooth(method="loess")
~~~
![6](https://user-images.githubusercontent.com/41892582/228041025-b9d088fe-74ab-4229-8795-2415a2a79856.jpg)

the relationship between quality and total sulfur dioxide 
~~~
ggplot(wine,aes(y=quality,x=total.sulfur.dioxide))+
  geom_smooth(method="loess")
~~~

![7](https://user-images.githubusercontent.com/41892582/228041039-166fc529-ef69-4c98-8a17-feada986c162.jpg)

the relationship between quality and density 
~~~
ggplot(wine,aes(y=quality,x=density))+
  geom_smooth(method="loess")
~~~
![8](https://user-images.githubusercontent.com/41892582/228041041-e2e4a73e-7032-46a2-97be-7026bef108fc.jpg)

the relationship between quality and pH 
~~~
ggplot(wine,aes(y=quality,x=pH))+
  geom_smooth(method="loess")
~~~
![9](https://user-images.githubusercontent.com/41892582/228041047-8ff301fa-78a0-4d2e-840d-6b4e7213c337.jpg)

the relationship between quality and sulphates 
~~~
ggplot(wine,aes(y=quality,x=sulphates))+
  geom_smooth(method="loess")
~~~
![10](https://user-images.githubusercontent.com/41892582/228041052-19d22c5f-c20e-459f-92de-80a07307f273.jpg)

the relationship between quality and alcohol 
~~~
ggplot(wine,aes(y=quality,x=alcohol))+
  geom_smooth(method="loess")
~~~

![11](https://user-images.githubusercontent.com/41892582/228041059-e247d18c-26cf-41e4-b8eb-93cd590b2688.jpg)



**Exploration of the data**

Let's see the correlation matrix of the dataset

~~~
corrplot(cor(train.wine), method="number")
~~~

![plot1](https://user-images.githubusercontent.com/41892582/227974685-94cb83d5-b18a-4141-9809-959ec1ef7fa6.png)

As we can see:

1-citric.acid and fixed.acidity are highly correlated

2-density and fixed.acidity are highly correlated

3-total.sulfur.dioxide and free.sulfur.dioxide are highly correlated

4-fixed.acidity and pH are highly correlated

5-citric.acid and volatile.acidity are highly correlated

6-citric.acid and pH are highly correlated

Calculate the partial correlation coefficient, t-statistic, and corresponding p-value. 

~~~
pcor(wine,method="pearson")
~~~

~~~
$estimate
                     fixed.acidity volatile.acidity  citric.acid residual.sugar    chlorides free.sulfur.dioxide total.sulfur.dioxide      density
fixed.acidity           1.00000000       0.05381063  0.333782406   -0.431331287 -0.229140937          0.10874458          -0.22749787  0.785552653
volatile.acidity        0.05381063       1.00000000 -0.529015039   -0.017443418  0.231549508         -0.15008012           0.18310398  0.120726024
citric.acid             0.33378241      -0.52901504  1.000000000    0.054222377  0.258467185         -0.16063844           0.25970894  0.007945372
residual.sugar         -0.43133129      -0.01744342  0.054222377    1.000000000 -0.007728558          0.12491066           0.03911292  0.595412132
chlorides              -0.22914094       0.23154951  0.258467185   -0.007728558  1.000000000          0.06370181          -0.15089356  0.090885165
free.sulfur.dioxide     0.10874458      -0.15008012 -0.160638440    0.124910661  0.063701811          1.00000000           0.66357314 -0.084264136
total.sulfur.dioxide   -0.22749787       0.18310398  0.259708936    0.039112917 -0.150893558          0.66357314           1.00000000  0.075954054
density                 0.78555265       0.12072602  0.007945372    0.595412132  0.090885165         -0.08426414           0.07595405  1.000000000
pH                     -0.71593146       0.01410286 -0.035855288   -0.326966596 -0.235310887          0.14000644          -0.19845101  0.570148248
sulphates              -0.15379695      -0.15710238  0.032157368   -0.202101953  0.363474405          0.04290040           0.05424922  0.247159448
alcohol                 0.51942235       0.12998418  0.152406331    0.478609975 -0.060535243         -0.03801112          -0.05190740 -0.725761311
quality                 0.02416842      -0.21914942 -0.031122793    0.027316022 -0.111508726          0.05035716          -0.11174916 -0.020744124
                              pH   sulphates     alcohol     quality
fixed.acidity        -0.71593146 -0.15379695  0.51942235  0.02416842
volatile.acidity      0.01410286 -0.15710238  0.12998418 -0.21914942
citric.acid          -0.03585529  0.03215737  0.15240633 -0.03112279
residual.sugar       -0.32696660 -0.20210195  0.47860997  0.02731602
chlorides            -0.23531089  0.36347441 -0.06053524 -0.11150873
free.sulfur.dioxide   0.14000644  0.04290040 -0.03801112  0.05035716
total.sulfur.dioxide -0.19845101  0.05424922 -0.05190740 -0.11174916
density               0.57014825  0.24715945 -0.72576131 -0.02074412
pH                    1.00000000 -0.11521819  0.51605787 -0.05411548
sulphates            -0.11521819  1.00000000  0.22393944  0.19722492
alcohol               0.51605787  0.22393944  1.00000000  0.25325645
quality              -0.05411548  0.19722492  0.25325645  1.00000000

$p.value
                     fixed.acidity volatile.acidity   citric.acid residual.sugar    chlorides free.sulfur.dioxide total.sulfur.dioxide       density
fixed.acidity         0.000000e+00     3.196249e-02  1.175642e-42   5.324605e-73 2.240090e-20        1.397317e-05         4.227637e-20  0.000000e+00
volatile.acidity      3.196249e-02     0.000000e+00 2.749350e-115   4.871557e-01 8.750158e-21        1.833553e-09         1.902648e-13  1.391019e-06
citric.acid           1.175642e-42    2.749350e-115  0.000000e+00   3.067096e-02 1.144028e-25        1.192855e-10         6.587145e-26  7.516411e-01
residual.sugar        5.324605e-73     4.871557e-01  3.067096e-02   0.000000e+00 7.582041e-01        5.887327e-07         1.191151e-01 4.683912e-153
chlorides             2.240090e-20     8.750158e-21  1.144028e-25   7.582041e-01 0.000000e+00        1.108905e-02         1.495332e-09  2.861269e-04
free.sulfur.dioxide   1.397317e-05     1.833553e-09  1.192855e-10   5.887327e-07 1.108905e-02        0.000000e+00        2.909593e-202  7.729694e-04
total.sulfur.dioxide  4.227637e-20     1.902648e-13  6.587145e-26   1.191151e-01 1.495332e-09       2.909593e-202         0.000000e+00  2.448008e-03
density               0.000000e+00     1.391019e-06  7.516411e-01  4.683912e-153 2.861269e-04        7.729694e-04         2.448008e-03  0.000000e+00
pH                   6.507469e-250     5.742812e-01  1.531171e-01   6.626524e-41 1.973061e-21        2.092132e-08         1.416478e-15 1.154336e-137
sulphates             7.157446e-10     3.041155e-10  2.001276e-01   4.154348e-16 8.121056e-51        8.734942e-02         3.058833e-02  1.522047e-23
alcohol              1.685061e-110     1.998697e-07  1.020416e-09   9.293088e-92 1.580506e-02        1.298828e-01         3.855427e-02 4.364101e-260
quality               3.356528e-01     9.872361e-19  2.149942e-01   2.764960e-01 8.373953e-06        4.474495e-02         8.004610e-06  4.086079e-01
                                pH    sulphates       alcohol      quality
fixed.acidity        6.507469e-250 7.157446e-10 1.685061e-110 3.356528e-01
volatile.acidity      5.742812e-01 3.041155e-10  1.998697e-07 9.872361e-19
citric.acid           1.531171e-01 2.001276e-01  1.020416e-09 2.149942e-01
residual.sugar        6.626524e-41 4.154348e-16  9.293088e-92 2.764960e-01
chlorides             1.973061e-21 8.121056e-51  1.580506e-02 8.373953e-06
free.sulfur.dioxide   2.092132e-08 8.734942e-02  1.298828e-01 4.474495e-02
total.sulfur.dioxide  1.416478e-15 3.058833e-02  3.855427e-02 8.004610e-06
density              1.154336e-137 1.522047e-23 4.364101e-260 4.086079e-01
pH                    0.000000e+00 4.132884e-06 7.407914e-109 3.100189e-02
sulphates             4.132884e-06 0.000000e+00  1.644730e-19 2.127228e-15
alcohol              7.407914e-109 1.644730e-19  0.000000e+00 1.123029e-24
quality               3.100189e-02 2.127228e-15  1.123029e-24 0.000000e+00

$statistic
                     fixed.acidity volatile.acidity citric.acid residual.sugar  chlorides free.sulfur.dioxide total.sulfur.dioxide     density         pH
fixed.acidity            0.0000000        2.1467733  14.1059183    -19.0458200 -9.3778399            4.357920            -9.306912  50.5728110 -40.850594
volatile.acidity         2.1467733        0.0000000 -24.8340233     -0.6950021  9.4819671           -6.047259             7.419786   4.8448186   0.561874
citric.acid             14.1059183      -24.8340233   0.0000000      2.1632483 10.6587860           -6.483581            10.713688   0.3165311  -1.429292
residual.sugar         -19.0458200       -0.6950021   2.1632483      0.0000000 -0.3078931            5.015367             1.559341  29.5231892 -13.782993
chlorides               -9.3778399        9.4819671  10.6587860     -0.3078931  0.0000000            2.542864            -6.080797   3.6356542  -9.644948
free.sulfur.dioxide      4.3579200       -6.0472594  -6.4835808      5.0153674  2.5428644            0.000000            35.335479  -3.3688260   5.632941
total.sulfur.dioxide    -9.3069116        7.4197864  10.7136881      1.5593411 -6.0807971           35.335479             0.000000   3.0345603  -8.066156
density                 50.5728110        4.8448186   0.3165311     29.5231892  3.6356542           -3.368826             3.034560   0.0000000  27.646909
pH                     -40.8505938        0.5618740  -1.4292922    -13.7829933 -9.6449481            5.632941            -8.066156  27.6469086   0.000000
sulphates               -6.2006071       -6.3372075   1.2817214     -8.2208102 15.5428576            1.710605             2.164323  10.1613916  -4.620739
alcohol                 24.2151998        5.2225091   6.1432021     21.7151100 -2.4159834           -1.515351            -2.070635 -42.0269187  24.001161
quality                  0.9630827       -8.9478019  -1.2404449      1.0885992 -4.4700697            2.008635            -4.479830  -0.8265650  -2.158971
                     sulphates    alcohol    quality
fixed.acidity        -6.200607  24.215200  0.9630827
volatile.acidity     -6.337208   5.222509 -8.9478019
citric.acid           1.281721   6.143202 -1.2404449
residual.sugar       -8.220810  21.715110  1.0885992
chlorides            15.542858  -2.415983 -4.4700697
free.sulfur.dioxide   1.710605  -1.515351  2.0086353
total.sulfur.dioxide  2.164323  -2.070635 -4.4798298
density              10.161392 -42.026919 -0.8265650
pH                   -4.620739  24.001161 -2.1589710
sulphates             0.000000   9.153586  8.0142971
alcohol               9.153586   0.000000 10.4290143
quality               8.014297  10.429014  0.0000000

$n
[1] 1599

$gp
[1] 10

$method
[1] "pearson"
~~~

the correlation between citric.acid and fixed.acidity are highly significant, as is the correlation between density and fixed.acidity,
also between total.sulfur.dioxide and free.sulfur.dioxide ,
fixed.acidity and pH and citric.acid and volatile.acidity We can safely assume that the independent variable has a high degree of correlation.

and let's check the suitability for factor analysis.

~~~
##new dataset of all independent variable
wine2<-dplyr::select(wine,-quality)
data_matrix<-cor(wine2)
KMO(data_matrix)
~~~

~~~
── Kaiser-Meyer-Olkin criterion (KMO) ─────────────────────────

✖ The overall KMO value for your data is unacceptable.
  These data are not suitable for factor analysis.

  Overall: 0.432

  For each variable:
       fixed.acidity     volatile.acidity          citric.acid 
               0.449                0.522                0.697 
      residual.sugar            chlorides  free.sulfur.dioxide 
               0.205                0.465                0.484 
total.sulfur.dioxide              density                   pH 
               0.452                0.366                0.449 
           sulphates              alcohol 
               0.509                0.229 
~~~

Since MSA < 0.5, we can't run factor analysis on this data.

to deal with the multicollinearity we will use the stepwise regression.

**Multiple linear regression model using all the data**

~~~
fit1<-lm(quality~.,train.wine)
summary(fit1)
~~~

~~~
Call:
lm(formula = quality ~ ., data = train.wine)

Residuals:
     Min       1Q   Median       3Q      Max 
-2.71905 -0.37014 -0.05358  0.42557  2.01296 

Coefficients:
                       Estimate Std. Error t value Pr(>|t|)    
(Intercept)           4.175e+01  2.368e+01   1.763 0.078128 .  
fixed.acidity         4.903e-02  2.916e-02   1.682 0.092911 .  
volatile.acidity     -1.201e+00  1.353e-01  -8.877  < 2e-16 ***
citric.acid          -3.314e-01  1.638e-01  -2.024 0.043192 *  
residual.sugar        2.744e-02  1.606e-02   1.709 0.087722 .  
chlorides            -1.758e+00  4.652e-01  -3.779 0.000165 ***
free.sulfur.dioxide   3.327e-03  2.415e-03   1.378 0.168592    
total.sulfur.dioxide -2.831e-03  7.938e-04  -3.566 0.000377 ***
density              -3.817e+01  2.416e+01  -1.580 0.114384    
pH                   -2.504e-01  2.150e-01  -1.165 0.244417    
sulphates             9.994e-01  1.264e-01   7.907  5.8e-15 ***
alcohol               2.485e-01  2.944e-02   8.440  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.6426 on 1238 degrees of freedom
Multiple R-squared:  0.3621,	Adjusted R-squared:  0.3564 
F-statistic: 63.89 on 11 and 1238 DF,  p-value: < 2.2e-16
~~~

Only 6 of the 11 independent variables are significant at the 0.05 level of significance, according to the adjusted R-squared of 0.3564, which indicates that independent variables explain 36% of the variation of the dependent variable.

**stepwise regression**

~~~
fit.step<-step(fit1,direction="both")
~~~

~~~
Start:  AIC=-1093.48
quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
    chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
    density + pH + sulphates + alcohol

                       Df Sum of Sq    RSS     AIC
- pH                    1     0.560 511.84 -1094.1
- free.sulfur.dioxide   1     0.784 512.06 -1093.6
<none>                              511.28 -1093.5
- density               1     1.031 512.31 -1093.0
- fixed.acidity         1     1.168 512.45 -1092.6
- residual.sugar        1     1.206 512.48 -1092.5
- citric.acid           1     1.692 512.97 -1091.3
- total.sulfur.dioxide  1     5.251 516.53 -1082.7
- chlorides             1     5.899 517.18 -1081.1
- sulphates             1    25.821 537.10 -1033.9
- alcohol               1    29.416 540.69 -1025.5
- volatile.acidity      1    32.543 543.82 -1018.4

Step:  AIC=-1094.11
quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
    chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
    density + sulphates + alcohol

                       Df Sum of Sq    RSS     AIC
- free.sulfur.dioxide   1     0.610 512.45 -1094.6
<none>                              511.84 -1094.1
+ pH                    1     0.560 511.28 -1093.5
- citric.acid           1     1.654 513.49 -1092.1
- residual.sugar        1     2.026 513.86 -1091.2
- density               1     2.985 514.82 -1088.8
- total.sulfur.dioxide  1     4.761 516.60 -1084.5
- fixed.acidity         1     5.360 517.20 -1083.1
- chlorides             1     5.386 517.22 -1083.0
- sulphates             1    27.100 538.94 -1031.6
- volatile.acidity      1    33.133 544.97 -1017.7
- alcohol               1    34.649 546.49 -1014.2

Step:  AIC=-1094.62
quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
    chlorides + total.sulfur.dioxide + density + sulphates + 
    alcohol

                       Df Sum of Sq    RSS     AIC
<none>                              512.45 -1094.6
+ free.sulfur.dioxide   1     0.610 511.84 -1094.1
+ pH                    1     0.386 512.06 -1093.6
- citric.acid           1     2.064 514.51 -1091.6
- residual.sugar        1     2.279 514.73 -1091.1
- density               1     3.034 515.48 -1089.2
- total.sulfur.dioxide  1     4.854 517.30 -1084.8
- chlorides             1     5.259 517.71 -1083.9
- fixed.acidity         1     5.477 517.93 -1083.3
- sulphates             1    27.238 539.69 -1031.9
- alcohol               1    35.045 547.49 -1013.9
- volatile.acidity      1    35.435 547.88 -1013.0
~~~

pH and free sulfur dioxide were omitted from the stepwise regression.

~~~
summary(fit.step)
~~~

~~~
Call:
lm(formula = quality ~ fixed.acidity + volatile.acidity + citric.acid + 
    residual.sugar + chlorides + total.sulfur.dioxide + density + 
    sulphates + alcohol, data = train.wine)

Residuals:
     Min       1Q   Median       3Q      Max 
-2.71483 -0.36547 -0.04754  0.41986  1.95999 

Coefficients:
                       Estimate Std. Error t value Pr(>|t|)    
(Intercept)           5.698e+01  2.000e+01   2.849 0.004451 ** 
fixed.acidity         7.410e-02  2.036e-02   3.640 0.000284 ***
volatile.acidity     -1.235e+00  1.334e-01  -9.260  < 2e-16 ***
citric.acid          -3.609e-01  1.615e-01  -2.235 0.025593 *  
residual.sugar        3.545e-02  1.510e-02   2.348 0.019021 *  
chlorides            -1.618e+00  4.536e-01  -3.567 0.000374 ***
total.sulfur.dioxide -2.021e-03  5.897e-04  -3.427 0.000630 ***
density              -5.432e+01  2.005e+01  -2.710 0.006829 ** 
sulphates             1.019e+00  1.255e-01   8.118 1.13e-15 ***
alcohol               2.319e-01  2.518e-02   9.209  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.6429 on 1240 degrees of freedom
Multiple R-squared:  0.3606,	Adjusted R-squared:  0.356 
F-statistic: 77.72 on 9 and 1240 DF,  p-value: < 2.2e-16
~~~

and all nine independent variables are significant in the model.

using the model to predict the quality from the testing data and compare it with the real data

~~~
test.step<-predict(fit.step,test.wine)   ##apply the model to the test data
test.step<-round(test.step)              ##round the value to the nearest integer
table(test.step,test.wine$quality)       ##compare the model's output to the actual data
~~~

~~~
test.step   3   4   5   6   7   8
        4   0   1   1   0   0   0
        5   1   6 102  32   1   0
        6   0   6  45  96  39   5
        7   0   0   0   5   7   1
        8   0   0   1   0   0   0
~~~

There are 206 estimated values that match the real data exactly, 129 that are shifted higher or lower by one unit, and 14 that are shifted higher or lower by two units.

**polynomial regression model**

In order to develop a polynomial regression model, we will add higher dimensional variables using the plots from the "Using Variables to Investigate Trends" section.

~~~
fit.poly<-glm(quality~poly(fixed.acidity,3)+
               volatile.acidity+
               poly(citric.acid,3)+
               residual.sugar+
               chlorides+
               free.sulfur.dioxide+
               poly(pH,3)+
               poly(total.sulfur.dioxide,3)+
               poly(density,2)+
               poly(sulphates,3)+
               poly(alcohol,3),data = train.wine)
~~~

~~~
summary(fit.poly)
~~~

~~~
Call:
glm(formula = quality ~ poly(fixed.acidity, 3) + volatile.acidity + 
    poly(citric.acid, 3) + residual.sugar + chlorides + free.sulfur.dioxide + 
    poly(pH, 3) + poly(total.sulfur.dioxide, 3) + poly(density, 
    2) + poly(sulphates, 3) + poly(alcohol, 3), data = train.wine)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.75611  -0.40222  -0.01106   0.42045   1.94856  

Coefficients:
                                Estimate Std. Error t value Pr(>|t|)    
(Intercept)                     6.229562   0.100807  61.797  < 2e-16 ***
poly(fixed.acidity, 3)1         2.994939   1.815889   1.649  0.09934 .  
poly(fixed.acidity, 3)2        -1.315777   0.894531  -1.471  0.14157    
poly(fixed.acidity, 3)3         0.450534   0.733397   0.614  0.53912    
volatile.acidity               -1.011315   0.137250  -7.368 3.17e-13 ***
poly(citric.acid, 3)1          -2.396827   1.124727  -2.131  0.03329 *  
poly(citric.acid, 3)2           1.136167   0.731352   1.554  0.12056    
poly(citric.acid, 3)3           0.815551   0.677448   1.204  0.22888    
residual.sugar                  0.022500   0.016645   1.352  0.17671    
chlorides                      -1.583939   0.484381  -3.270  0.00111 ** 
free.sulfur.dioxide             0.001046   0.002599   0.403  0.68724    
poly(pH, 3)1                   -2.232415   1.219254  -1.831  0.06735 .  
poly(pH, 3)2                   -1.033211   0.783466  -1.319  0.18749    
poly(pH, 3)3                    0.622790   0.679084   0.917  0.35927    
poly(total.sulfur.dioxide, 3)1 -2.750221   1.010317  -2.722  0.00658 ** 
poly(total.sulfur.dioxide, 3)2  0.419216   0.724484   0.579  0.56294    
poly(total.sulfur.dioxide, 3)3  2.133553   0.660471   3.230  0.00127 ** 
poly(density, 2)1              -4.477622   1.651432  -2.711  0.00679 ** 
poly(density, 2)2               1.384825   0.871425   1.589  0.11228    
poly(sulphates, 3)1             6.856161   0.757482   9.051  < 2e-16 ***
poly(sulphates, 3)2            -4.222239   0.712036  -5.930 3.94e-09 ***
poly(sulphates, 3)3             2.932886   0.677213   4.331 1.61e-05 ***
poly(alcohol, 3)1               7.548177   1.143757   6.599 6.13e-11 ***
poly(alcohol, 3)2               0.508906   0.704594   0.722  0.47027    
poly(alcohol, 3)3              -0.538546   0.660758  -0.815  0.41521    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.3911509)

    Null deviance: 801.50  on 1249  degrees of freedom
Residual deviance: 479.16  on 1225  degrees of freedom
AIC: 2400.8

Number of Fisher Scoring iterations: 2
~~~

using the model to predict the quality from the testing data and compare it with the real data

~~~
test.poly<-predict(fit.poly,test.wine)   ##apply the model to the test data
test.poly<-round(test.poly)              ##round the value to the nearest integer
table(test.poly,test.wine$quality)       ##compare the model's output to the actual data
~~~

~~~
test.poly   3   4   5   6   7   8
        4   0   1   0   0   0   0
        5   1   5 102  31   1   0
        6   0   7  46  97  34   4
        7   0   0   1   5  12   2
~~~

There are 212 estimated values that match the real data exactly, 123 that are shifted higher or lower by one unit, and 14 that are shifted higher or lower by two units.

**Spline Regression**

We'll examine whether or not adding splines to the polynomial model will make it better.

~~~
fit.spline<-glm(quality~ns(fixed.acidity,3)+
               volatile.acidity+
               ns(citric.acid,3)+
               residual.sugar+
               chlorides+
               ns(total.sulfur.dioxide,3)+
               ns(density,2)+
               ns(sulphates,3)+
               ns(alcohol,3),data = train.wine)
~~~

~~~
summary(fit.spline)
~~~
~~~
Call:
glm(formula = quality ~ ns(fixed.acidity, 3) + volatile.acidity + 
    ns(citric.acid, 3) + residual.sugar + chlorides + free.sulfur.dioxide + 
    poly(pH, 3) + ns(total.sulfur.dioxide, 3) + ns(density, 2) + 
    ns(sulphates, 3) + ns(alcohol, 3), data = train.wine)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.71909  -0.40321  -0.00618   0.40696   1.93381  

Coefficients:
                              Estimate Std. Error t value Pr(>|t|)    
(Intercept)                   5.049388   0.318827  15.837  < 2e-16 ***
ns(fixed.acidity, 3)1         0.474649   0.193933   2.447 0.014525 *  
ns(fixed.acidity, 3)2         1.204143   0.490939   2.453 0.014316 *  
ns(fixed.acidity, 3)3         0.475752   0.352065   1.351 0.176843    
volatile.acidity             -1.036878   0.138008  -7.513 1.11e-13 ***
ns(citric.acid, 3)1          -0.266649   0.114980  -2.319 0.020554 *  
ns(citric.acid, 3)2          -0.168821   0.162988  -1.036 0.300505    
ns(citric.acid, 3)3          -0.031403   0.158393  -0.198 0.842877    
residual.sugar                0.030300   0.016660   1.819 0.069208 .  
chlorides                    -1.830347   0.473165  -3.868 0.000115 ***
free.sulfur.dioxide           0.001243   0.002621   0.474 0.635421    
poly(pH, 3)1                 -1.843956   1.225357  -1.505 0.132625    
poly(pH, 3)2                 -0.600669   0.787186  -0.763 0.445576    
poly(pH, 3)3                  0.345714   0.677197   0.511 0.609788    
ns(total.sulfur.dioxide, 3)1 -0.474038   0.156728  -3.025 0.002542 ** 
ns(total.sulfur.dioxide, 3)2  0.058844   0.271412   0.217 0.828395    
ns(total.sulfur.dioxide, 3)3 -0.020001   0.411234  -0.049 0.961218    
ns(density, 2)1              -1.580961   0.484717  -3.262 0.001138 ** 
ns(density, 2)2              -0.562666   0.280846  -2.003 0.045347 *  
ns(sulphates, 3)1             1.377688   0.136445  10.097  < 2e-16 ***
ns(sulphates, 3)2             2.180940   0.364191   5.988 2.78e-09 ***
ns(sulphates, 3)3             0.574931   0.350121   1.642 0.100828    
ns(alcohol, 3)1               0.471201   0.108327   4.350 1.48e-05 ***
ns(alcohol, 3)2               1.108694   0.341483   3.247 0.001199 ** 
ns(alcohol, 3)3               1.113397   0.201616   5.522 4.08e-08 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.393641)

    Null deviance: 801.50  on 1249  degrees of freedom
Residual deviance: 482.21  on 1225  degrees of freedom
AIC: 2408.7

Number of Fisher Scoring iterations: 2

~~~

using the model to predict the quality from the testing data and compare it with the real data

~~~
test.spline<-predict(fit.spline,test.wine)   ##apply the model to the test data
test.spline<-round(test.spline)              ##round the value to the nearest integer
table(test.spline,test.wine$quality)       ##compare the model's output to the actual data
~~~

~~~
test.spline  3  4  5  6  7  8
          4  0  2  1  0  0  0
          5  1  5 97 30  1  0
          6  0  6 50 94 37  4
          7  0  0  1  9  9  2
~~~

There are 202 estimated values that match the real data exactly, 134 that are shifted higher or lower by one unit, and 13 that are shifted higher or lower by two units.

**Regression tree**

~~~
##the Regression tree model
dt<-rpart(quality~.,data =train.wine)
print(dt)
~~~

~~~
n= 1250 

node), split, n, deviance, yval
      * denotes terminal node

 1) root 1250 801.504800 5.629600  
   2) alcohol< 10.525 772 337.796600 5.379534  
     4) sulphates< 0.535 185  51.654050 5.043243 *
     5) sulphates>=0.535 587 258.626900 5.485520  
      10) volatile.acidity>=0.345 523 209.766700 5.424474  
        20) alcohol< 9.975 362 129.881200 5.328729 *
        21) alcohol>=9.975 161  69.105590 5.639752 *
      11) volatile.acidity< 0.345 64  30.984380 5.984375 *
   3) alcohol>=10.525 478 337.464400 6.033473  
     6) sulphates< 0.645 208 150.110600 5.668269  
      12) volatile.acidity>=1.015 10   6.000000 4.000000 *
      13) volatile.acidity< 1.015 198 114.873700 5.752525  
        26) volatile.acidity>=0.3025 175  89.428570 5.657143  
          52) free.sulfur.dioxide< 6.5 50  32.500000 5.300000 *
          53) free.sulfur.dioxide>=6.5 125  48.000000 5.800000 *
        27) volatile.acidity< 0.3025 23  11.739130 6.478261 *
     7) sulphates>=0.645 270 138.240700 6.314815  
      14) alcohol< 11.65 174  78.465520 6.120690  
        28) volatile.acidity>=0.395 93  27.612900 5.935484 *
        29) volatile.acidity< 0.395 81  44.000000 6.333333  
          58) pH>=3.255 50  24.820000 6.060000 *
          59) pH< 3.255 31   9.419355 6.774194 *
      15) alcohol>=11.65 96  41.333330 6.666667 *
~~~

~~~
##the plot of the decision tree model
rpart.plot(dt)
~~~

![dt](https://user-images.githubusercontent.com/41892582/228220685-05c4736f-c2d0-4a69-b161-06ac4955086d.jpg)

using the model to predict the quality from the testing data and compare it with the real data

~~~
test.dt<-predict(dt,test.wine)   ##apply the model to the test data
test.dt<-round(test.dt)          ##round the value to the nearest integer
table(test.dt,test.wine$quality) ##compare the model's output to the actual data
~~~

~~~
test.dt   3   4   5   6   7   8
      5   1   6 100  41   8   0
      6   0   7  48  78  21   1
      7   0   0   1  14  18   5
~~~~

There are 196 estimated values that match the real data exactly, 135 that are shifted higher or lower by one unit, and 18 that are shifted higher or lower by two units.


**Random forest**

~~~
rf<-randomForest(quality~.,ntree=500,data =train.wine)
plot(rf)
~~~

![rf](https://user-images.githubusercontent.com/41892582/228222777-cd8d0731-882c-4089-aaf3-061b370a991d.jpg)

~~~
print(rf)
Call:
 randomForest(formula = quality ~ ., data = train.wine, ntree = 500) 
               Type of random forest: regression
                     Number of trees: 500
No. of variables tried at each split: 3

          Mean of squared residuals: 0.3321053
                    % Var explained: 48.21
~~~

Mean of  Square Residals: 0.33 We need to know if, if we add more trees, the Mean of squared residuals would change or not.

~~~
rf2<-randomForest(quality~.,ntree=1000,data =train.wine)
plot(rf2)
~~~

![rf2](https://user-images.githubusercontent.com/41892582/228225016-131e353a-63e2-4ecb-a92e-e8049cca4fa6.jpg)

~~~
print(rf2)

Call:
 randomForest(formula = quality ~ ., data = train.wine, ntree = 1000) 
               Type of random forest: regression
                     Number of trees: 1000
No. of variables tried at each split: 3

          Mean of squared residuals: 0.3279189
                    % Var explained: 48.86
~~~

When we add more trees, the Mean of Square residuals doesn't change much, as well as the error rate and it seems to be constant after 200 trees.

using the model to predict the quality from the testing data and compare it with the real data

~~~
test.rf2<-predict(rf2,test.wine)   ##apply the model to the test data
test.rf2<-round(test.rf2)         ##round the value to the nearest integer
table(test.rf2,test.wine$quality) ##compare the model's output to the actual data
~~~

~~~
test.rf2   3   4   5   6   7   8
       5   1   9 115  25   3   0
       6   0   4  34 104  30   2
       7   0   0   0   4  14   4
~~~

There are 233 estimated values that match the real data exactly, 106 that are shifted higher or lower by one unit, and 10 that are shifted higher or lower by two units.
