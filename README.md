# AmazonProductRecSystem_MIT2025
Amazon Product Recommendation System

Author: Austin McCollough  
Date: June 2025  
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-surprise

Table of Contents

Project Overview
Dataset
Methodology
  1. Rank-Based Recommendations
  2. Collaborative Filtering
  3. Matrix Factorization (SVD)
  4. Hyperparameter Tuning
Evaluation Metrics
Results
Key Takeaways
Example Usage
Business Value
Next Steps

Project Overview

This project implements a robust recommendation engine for Amazon’s electronics catalog using collaborative filtering and matrix factorization. The goal is to help users discover products tailored to their preferences, leveraging historical ratings and advanced algorithms.

Dataset

Source: Amazon Electronics ratings dataset
Preprocessing:
  Users with ≥50 ratings, products with ≥5 ratings
  Final dataset: 65,290 ratings, 1,540 users, 5,689 products
  Columns: user_id, prod_id, rating
  No missing values; data types validated

Methodology

1. Rank-Based Recommendations

Recommend top-rated products with sufficient interactions
Useful as a baseline and for cold-start scenarios

2. Collaborative Filtering

User-User Similarity (KNN, cosine/Pearson): Finds users with similar rating patterns
Item-Item Similarity (KNN, cosine/MSD): Recommends products similar to those a user already likes

3. Matrix Factorization (SVD)

Captures latent user/item features for personalized recommendations
Outperformed other models on all key metrics

4. Hyperparameter Tuning

Used GridSearchCV to optimize model parameters for best RMSE, precision, recall, and F1 score

Evaluation Metrics

RMSE: Root Mean Squared Error (lower = better)
Precision@10, Recall@10, F1 Score: Evaluates recommendation quality for top-10 suggestions per user

Results

| Model                | RMSE   | Precision@10 | Recall@10 | F1 Score |
|----------------------|--------|--------------|-----------|----------|
| User-User (KNN)      | 1.0363 | 0.847        | 0.878     | 0.862    |
| Item-Item (KNN)      | 0.9578 | 0.839        | 0.880     | 0.859    |
| SVD (Optimized)  | 0.8822 | 0.854        | 0.884     | 0.869    |

SVD outperformed all other models in both accuracy and recommendation quality.

Key Takeaways

Optimized SVD is the recommended algorithm for production deployment
Regular retraining and tuning are essential as user behavior and product inventory evolve
For new users/products, blend in popularity-based or content-based recommendations to address cold start
Incorporate additional product/user features to further enhance personalization
Introduce diversity in recommendations to prevent user fatigue

Example Usage

from surprise import SVD, Dataset, Reader

Prepare data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_final['user_id', 'prod_id', 'rating']], reader)
trainset = data.build_full_trainset()

Train optimized SVD model
svd = SVD(n_epochs=20, lr_all=0.01, reg_all=0.4, random_state=1)
svd.fit(trainset)

Predict rating for a user and product
svd.predict("A3LDPF5FMB782Z", "1400501466")

Business Value

Deploying a well-tuned SVD model enables Amazon to deliver highly relevant product suggestions, driving user engagement and increasing sales. The approach is scalable, interpretable, and adaptable to changing user preferences.

Next Steps

Integrate the model into marketing and on-site personalization workflows
Monitor performance and retrain periodically
Expand with product metadata and user features for even better personalization

Notebook:  
See 2025_FINAL_A.McCollough_Amazon_Product_Recommendation.ipynb for full code and analysis.
https://drive.google.com/file/d/134KUakdgFgAF7UPI-MEkTlk6Ofv11CfO/view?usp=share_link

Want to collaborate or have questions? Reach out via [GitHub] or email.

GH: https://github.com/amccolloughdatascience

E: amcco.datascience@gmail.com

