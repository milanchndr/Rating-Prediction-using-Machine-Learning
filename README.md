# Rating-Prediction-using-Machine-Learning

This project aims to predict the rating of recipe reviews based on various features such as user reputation, thumbs up/down counts, and the textual content of the review itself. The dataset used for this project is from a Kaggle competition.

## Dataset

The dataset consists of two files:

1. `train.csv`: This file contains the training data with the following columns:
   - `ID`: Unique identifier for each row
   - `RecipeCode`: Unique identifier for each recipe
   - `RecipeNumber`: Another unique identifier for each recipe
   - `UserID`: Unique identifier for each user
   - `UserReputation`: Reputation score of the user
   - `CommentID`: Unique identifier for each comment
   - `RecipeName`: Name of the recipe
   - `Recipe_Review`: Text content of the recipe review
   - `CreationTimestamp`: Time when the review was created (in Unix timestamp format)
   - `ReplyCount`: Number of replies to the review
   - `ThumbsUpCount`: Number of thumbs up received for the review
   - `ThumbsDownCount`: Number of thumbs down received for the review
   - `BestScore`: Best score received for the review
   - `Rating`: Target variable (rating of the review)

2. `test.csv`: This file contains the test data with the same columns as `train.csv`, except for the `Rating` column.

## Exploratory Data Analysis

The notebook starts with an exploratory data analysis (EDA) to gain insights into the dataset. Some key observations from the EDA are:

- The numerical features `UserReputation`, `ThumbsUpCount`, and `ThumbsDownCount` show some correlation with the `Rating` column.
- The `Rating` distribution is highly skewed.
- No significant patterns were found when analyzing the relationship between rating and time-related features (month, day, hour).

## Data Preprocessing

Based on the EDA findings, the following preprocessing steps were performed:

1. Dropping irrelevant columns (`ReplyCount`, `BestScore`, and time-related columns).
2. Handling missing values by imputing with the most frequent value.
3. Text preprocessing steps:
   - Converting text to lowercase
   - Decontracting words (e.g., "won't" to "will not")
   - Removing special characters, stopwords, and multiple spaces
4. Removing outliers based on z-score for numerical features and word counts in the `Recipe_Review` column.
5. Scaling numerical features using `StandardScaler`.
6. Vectorizing text features using `TfidfVectorizer`.

## Modeling

Eight different models were trained and evaluated on the preprocessed data:

1. Logistic Regression
2. Random Forest
3. LightGBM
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Decision Tree
7. Gradient Boosting
8. Multi-layer Perceptron (MLP)

The performance of each model was compared based on accuracy, precision, F1-score, and classification report.

## Hyperparameter Tuning

The top three best-performing models (Logistic Regression, LightGBM, and Random Forest) were further tuned using hyperparameter tuning to improve their performance.

## Results

The best-performing model after hyperparameter tuning was LightGBM, achieving an accuracy of 79.16%.

## Usage

To run the code and reproduce the results, follow these steps:

1. Clone the repository: `git clone https://github.com/your-repo-url.git`
2. Install the required Python packages: `pip install -r requirements.txt`
3. Download the dataset from the Kaggle competition and place the `train.csv` and `test.csv` files in the project directory.
4. Open the Jupyter Notebook file (`Rating Prediction Notebook .ipynb`) and run the cells.

Note: Make sure to update the file paths in the code if you place the dataset files in a different location.

## Future Work

Some potential areas for future improvement include:

- Exploring more advanced text preprocessing techniques, such as word embeddings or language models.
- Implementing ensemble methods or stacking to combine the strengths of multiple models.
- Conducting more extensive hyperparameter tuning and feature engineering.
- Investigating other evaluation metrics or techniques to handle the imbalanced class distribution.

## Acknowledgments

This project was completed as part of the recipe-for-rating-predict-food-ratings-using-ml. Special thanks to the IIT Madras and the Kaggle community for providing the dataset and resources.


