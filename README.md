# M2R: Random Forests

## Overview
This project
  - studied and implemented Decision Trees and Random Forests for     classification problems
  - performed simulations on various [datasets](#datasets)

The [example code](https://github.com/liyiyan128/M2R-random-forests/blob/main/example_code_iris.ipynb) applies DecisionTree and RandomForest to the Iris dataset.

**Summary**
- <ins>RF accuracy</ins> depends on individual tree *strength* and *correlation*
- LLN guarantees the generalisation error converges a.s. as `n_trees` increases,<br> so that overfitting is not a problem.
- <ins>RF performance</ins> is *insensitive* to `m_features`,<br> and usually `m_features=1` or `m_features=2` gives near optimum results   

## Contents
- [M2R: Random Forests](#m2r-random-forests)
  - [Overview](#overview)
  - [Contents](#contents)
  - [Simulations](#simulations)
  - [Datasets](#datasets)
  - [Reference](#reference)

## Simulations
[pdf](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations.pdf)/[ipynb](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/simulations.ipynb)
- [RandomForest Performance as n_trees Varies](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/RandomForest-Performance-as-n_trees-Varies.png)
- [ImprovedRandomForest Performance as n_trees Varies](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/ImprovedRandomForest-Performance-as-n_trees-Varies.png)
- [RandomForest Performance on the Titanic Dataset](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/RandomForest-Performance-on-the-Titanic-Dataset.png)

## Datasets
- [Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [Wine](https://archive.ics.uci.edu/dataset/109/wine)
- [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
- [Breast Cancer](https://archive.ics.uci.edu/dataset/14/breast+cancer)
- [Titanic](https://www.kaggle.com/competitions/titanic/data)

## Reference
[1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning: with Applications in R. Springer.](https://github.com/liyiyan128/M2R-random-forests/blob/main/materials/An%20Introduction%20to%20Statistical%20Learning.pdf)

[2. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.](https://github.com/liyiyan128/M2R-random-forests/blob/main/materials/Breiman.pdf)