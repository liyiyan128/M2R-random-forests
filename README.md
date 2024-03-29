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

![Fig 1. RF Performance as n_trees Varies](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/RF-n50.png)

![Fig 2. RF Performance as m_features Varies](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/RF-Titanic-m4.png)

![Fig 3. RF & ImprovedRF Performance on the Titanic Dataset](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/RF-IRF-Titanic.png)

![Fig 4. RF & ImprovedRF Performance on the Breast Cancer Dataset](https://github.com/liyiyan128/M2R-random-forests/blob/main/simulations/graphs/RF-IRF-Performance-on-the-Breast-Cancer-Dataset.png)

## Datasets
- [Iris](https://archive.ics.uci.edu/dataset/53/iris): $150 \times 4$, continuous, simple
- [Wine](https://archive.ics.uci.edu/dataset/109/wine): $178 \times 13$, continuous, simple
- [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom): $8124 \times 22$ (clean), categorical, simple
- [Breast Cancer](https://archive.ics.uci.edu/dataset/14/breast+cancer): $277 \times 9$ (clean), categorical
- [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease): $297 \times 13$ (clean), complex
- [Titanic](https://www.kaggle.com/competitions/titanic/data): $712 \times 7$ (clean), complex

## Reference

[[1] Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.](https://github.com/liyiyan128/M2R-random-forests/blob/main/materials/Breiman.pdf)

[[2] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning: with Applications in R. Springer.](https://github.com/liyiyan128/M2R-random-forests/blob/main/materials/An%20Introduction%20to%20Statistical%20Learning.pdf)
