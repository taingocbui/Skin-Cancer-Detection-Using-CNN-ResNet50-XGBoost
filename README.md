# Collaborative Filtering Recommendation System

Author: Tai Ngoc Bui <br>
Date Completion: December 7th, 2024

## 1.Business Understanding
This project focuses on developing a robust Convolutional Neural Network (CNN) capable of classifying skin lesions as malignant or benign. Skin cancer is one of the most prevalent cancers worldwide, and early detection significantly improves treatment outcomes and survival rates. Despite advancements in medical technology, many cases are either detected too late or misdiagnosed due to the limitations of traditional diagnostic methods. By leveraging deep learning techniques, I aim to provide a reliable, efficient, and scalable solution to assist dermatologists and healthcare professionals in diagnosing skin cancer.

This topic is personally significant as it combines two areas of great interest: leveraging artificial intelligence to solve real-world problems and contributing to public health initiatives. On a broader level, this research has societal implications, potentially saving lives and reducing healthcare costs. The target audience for this project includes not only potential patients of skin cancer but also dermatologists and oncologists seeking diagnostic support tools, as well as the whole healthcare system. To successfully complete this project, I relied on numerous studies and projects highlighting the performance of CNNs like ResNet, InceptionNet and MobileNet for skin cancer detection. Moreover, studies on transfer learning to improve model performance with limited datasets are also utilized in this project.

This model will use recall rate as the main metric to prioritize the ability of the model to correctly identify all positive cases (e.g. malignant cancer). In medical diagnoses, a false negative (failing to detect cancer when it is present) can have severe consequences, such as delayed treatment or worsened prognosis. Recall ensures that the model minimizes false negatives, even if it occasionally produces false positives. Missing a malignant case is riskier than flagging a benign case as malignant, as false positives can often be corrected through follow-up procedures.


## 2.Data Understanding
The dataset used in this project is sourced from Kaggle’s [Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign). It consists of two folders: train and test with images (224x244) of the two types of moles including benign skin moles and malignant skin moles. It is curated for machine learning research and adheres to ethical standards for data use. The primary features include images of skin lesions data and categorical labels indicating whether the lesion is benign or malignant. Several researchers have used this dataset to train and evaluate machine learning models for skin cancer detection. 

## 3. Objectives
Develop a robust Convolutional Neural Network (CNN) capable of classifying skin lesions as malignant or benign using skin images. This classifying model should achieve a recall rate at least 80% to correctly identify all positive cases.

## 4. Exploratory Analysis
The data is stored as jpg image files and already separated into train and test folder. Each image (224x244) represents a skin lesion and is a collection of pixel values, typically stored as arrays of integers ranging from 0 to 255 (grayscale) or three-channel RGB values. The labels are categorical, indicating whether the lesion is benign (label 0) or malignant (label 1). 
![Train Samples](https://github.com/taingocbui/phase5_project/blob/main/photos/Train_samples.png)
![Test Samples](https://github.com/taingocbui/phase5_project/blob/main/photos/Test_samples.png)






### a. Base Model (KNNBasic)
To start with, I use KNNBasic as the baseline model with both cosine and pearson similarity function. It is easy to understand and implement. It uses the k-nearest neighbors approach to find similar users or items based on interaction data. Both model had RMSE of about 0.97, meaning that it was off by almost 1 point for each guess it made for ratings.

Cosine similarity and Pearson similarity are both metrics for measuring similarity but differ in focus and application. Cosine similarity measures the cosine of the angle between two vectors, focusing on their direction while ignoring magnitude, making it ideal for comparing high-dimensional or sparse data where the scale is irrelevant. In contrast, Pearson similarity evaluates the linear correlation between two vectors, accounting for their mean and relative deviations, making it suitable for analyzing relationships or trends in the data. While cosine similarity emphasizes vector orientation, Pearson similarity captures how well one variable's variations predict another’s, regardless of scaling or translation.

### b. KNNBaseline
To further improve the neighborhood-based model, I will apply the KNNBaseline method, adding in bias term to reduce overfitting. KNNBasic directly computes similarity between users or items based on raw ratings, making it simple and intuitive but less effective in datasets with systematic biases, such as users who consistently rate items higher or lower than average. On the other hand, KNNBaseline accounts for these biases by incorporating baseline estimates for ratings, which adjust for global, user-specific, and item-specific biases. By adding in bias term, KNNBaseline method achieved the best RMSE of 0.87 with both Pearson and Cosine similarity.

Overall, Memory-based Collaborative Filtering is easy to implement and produce reasonable prediction quality. However, there are some drawback of this approach:

- It doesn't address the well-known cold-start problem, that is when new user or new item enters the system.
- It can't deal with sparse data, meaning it's hard to find users that have rated the same items.
- It suffers when new users or items that don't have any ratings enter the system.
- It tends to recommend popular items.

### c. Matrix Factorization (Singular Vector Decomposition)
Besides memory-based methods, model-based method is another popular approach of Collaborative Filtering. Unlike memory-based collaborative filtering, which directly calculates similarity scores between users or items (such as user-based or item-based nearest-neighbor algorithms), model-based approaches use a data-driven model to generalize patterns in the data.

In model-based collaborative filtering, a model is trained to learn hidden (latent) factors that represent users and items. These factors capture preferences in a reduced feature space. Techniques like matrix factorization (e.g., Singular Value Decomposition (SVD), Non-negative Matrix Factorization (NMF)) are popular for learning these latent factors, effectively representing users and items as vectors in this space.

Matrix factorization is widely used for recommender systems where it can deal better with scalability and sparsity than Memory-based CF. The goal of MF is to learn the latent preferences of users and the latent attributes of items from known ratings (learn features that describe the characteristics of ratings) to then predict the unknown ratings through the dot product of the latent features of users and

While model-based collaborative filtering excels in scalability and predictive power, its drawbacks include:

- challenges with new users/items (similar to memory-based model)
- high initial computational cost for the decomposition
- overfitting risks
- requires periodic retraining to incorporate new interactions

With this MovieLens dataset, SVD is the best model with the lowest test set root squared mean error of 0.8519. The parameters used for tuning SVD is as follows: 'lr_all': 0.005, 'n_factors': 250, 'reg_all': 0.05, 'n_epochs': 100

## 5. Conclusion
Based on our analysis and testing with different models, I want to recommend Model-based Singular Value Decomposition (SVD) with user-based focus as the final model for the recommendation system given the MovieLens dataset. The parameters used in this model is 'lr_all' = 0.005, 'n_factors' = 250, 'reg_all' = 0.05, and 'n_epochs' = 100. Based on our test results, this model can achieve 0.8519 root mean squared error.

Another reason for using Model-based SVD is its scalability for large scale applications. This scalability ensures that SVD-based systems can handle millions of users and items effectively. Moreover, SVD can uncover latent relationships between users and items, such as implicit preferences or themes, which are not immediately visible in the raw interaction matrix. These latent features improve the system's ability to predict user preferences, even for items that the user has not explicitly interacted with.

## 6. Future Works
To better improve the quality of this report, I will extend this project by investigate deep learning-based methods like Neural Collaborative Filtering (NCF) or autoencoders for better capture of complex user-item interactions. Another possibility to improve this project is to combine collaborative filtering with content-based filtering or context-aware techniques to address limitations like cold-start problems.


