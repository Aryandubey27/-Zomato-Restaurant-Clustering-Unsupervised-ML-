# Zomato Restaurant Clustering and Sentiment Analysis

## Project Overview

This project focuses on analyzing Zomato restaurant data for various cities in India to derive actionable insights into the Indian food industry. Utilizing unsupervised machine learning techniques, particularly clustering, the goal is to segment restaurants into distinct groups based on their characteristics and customer feedback. The project also involves extensive Exploratory Data Analysis (EDA) and sentiment analysis of user reviews to understand customer preferences and identify areas for business improvement for Zomato.

## Table of Contents

1.  [Business Problem](#business-problem)
2.  [Project Goal](#project-goal)
3.  [Dataset Information](#dataset-information)
4.  [Methodology](#methodology)
5.  [Key Findings & Insights](#key-findings--insights)
6.  [ML Models Used](#ml-models-used)
7.  [Evaluation & Business Impact](#evaluation--business-impact)
8.  [Tools & Technologies](#tools--technologies)
9.  [How to Run the Project](#how-to-run-the-project)
10. [Future Work](#future-work)
11. [Contribution](#contribution)
12. [License](#license)

## Business Problem

[cite_start]Zomato, an Indian restaurant aggregator and food delivery startup, needs to analyze its vast restaurant and user review data to gain insights into the Indian food industry across various cities[cite: 1]. [cite_start]The challenge is to effectively analyze customer sentiments, identify different restaurant segments through clustering, and derive actionable business conclusions that benefit both customers (by helping them find suitable restaurants) and the company (by identifying areas for growth and improvement)[cite: 1].

## Project Goal

The primary goal of this project is to analyze Zomato restaurant data to:
* [cite_start]Analyze the sentiments of the reviews given by the customer in the data[cite: 1].
* [cite_start]Cluster the Zomato restaurants into different segments[cite: 1].
* [cite_start]Visualize the data to easily analyze data at instant[cite: 1].
* [cite_start]Solve business cases that can directly help customers find the best restaurant in their locality[cite: 1].
* [cite_start]Help the company grow up and work on the fields they are currently lagging in[cite: 1].

## Dataset Information

The project utilizes two primary datasets, provided as CSV files:

1.  **`Zomato Restaurant names and Metadata.csv`**: Contains details about restaurants.
    * [cite_start]`Name`: Name of Restaurants [cite: 1]
    * [cite_start]`Links`: URL Links of Restaurants [cite: 1]
    * [cite_start]`Cost`: Per person estimated cost of dining [cite: 1]
    * `Collections`: Tagging of Restaurants w.r.t. [cite_start]Zomato categories [cite: 1]
    * [cite_start]`Cuisines`: Cuisines served by restaurants [cite: 1]

2.  **`Zomato Restaurant reviews.csv`**: Contains user review data.
    * `Restaurant`: Name of the Restaurant (implied link to `Name` in the restaurant data)
    * [cite_start]`Reviewer`: Name of the reviewer [cite: 1]
    * [cite_start]`Review`: Review text [cite: 1]
    * [cite_start]`Rating`: Rating provided [cite: 1]
    * [cite_start]`Metadata`: Reviewer metadata - No of reviews and followers [cite: 1]
    * [cite_start]`Time`: Date and Time of Review [cite: 1]
    * [cite_start]`Pictures`: No of pictures posted with review [cite: 1]

## Methodology

The project follows a structured Machine Learning pipeline:

1.  **Exploratory Data Analysis (EDA) & Visualization:**
    * [cite_start]Understanding business problem and data distribution[cite: 1].
    * [cite_start]Visualizing and analyzing relationships, forming assumptions, and obtaining insights[cite: 1].
    * [cite_start]Creating at least 15 logical and meaningful charts following the UBM (Univariate, Bivariate, Multivariate) rule[cite: 1].
2.  **Data Cleaning & Feature Engineering:**
    * [cite_start]Handling Missing Values and Outliers[cite: 1].
    * [cite_start]Categorical Encoding and Textual Data Preprocessing (if applicable)[cite: 1].
    * [cite_start]Creating new features and selecting important ones to avoid overfitting[cite: 1].
    * [cite_start]Data Transformation and Scaling[cite: 1].
    * [cite_start]Dimensionality Reduction (if needed)[cite: 1].
3.  **Hypothesis Testing:**
    * [cite_start]Defining and testing at least three hypothetical statements derived from dataset insights[cite: 1].
4.  **ML Model Implementation:**
    * [cite_start]Implementing and training appropriate clustering algorithms[cite: 1].
    * [cite_start]Performing Cross-Validation & Hyperparameter Tuning[cite: 1].
    * [cite_start]Evaluating models using relevant metrics[cite: 1].
5.  **Model Explainability:**
    * [cite_start]Analyzing cluster characteristics and "feature importance" to explain model insights[cite: 1].

## Key Findings & Insights

* **Market Concentration:** Most Zomato restaurants primarily cater to the budget-to-mid-range dining segments, indicating a competitive landscape in these areas.
* **Dominant Cuisines:** Cuisines like North Indian, Chinese, and Fast Food are highly prevalent and popular, reflecting strong market demand and supply.
* **High Customer Satisfaction:** The distribution of average ratings is skewed towards higher scores, indicating a generally positive perception of restaurants on the platform.
* **Review Behavior:** User review activity shows distinct patterns, with significant peaks observed during weekends, which aligns with typical leisure and dining habits.
* **Value-for-Money Trend:** While higher-cost restaurants generally correlate with higher ratings, a substantial number of moderately priced establishments also achieve excellent ratings, highlighting a strong "value-for-money" segment.
* **Actionable Feedback:** Sentiment analysis through word clouds identified "food," "service," and "ambiance" as key drivers of positive reviews, whereas "delivery," "wait times," and "cold food" were common themes in negative feedback, providing direct actionable areas for improvement for Zomato and its restaurant partners.
* **Restaurant Segmentation:** The K-Means clustering model successfully identified distinct restaurant segments (e.g., [Your Optimal K value, example: 4-5 clusters]). These clusters are characterized by unique profiles based on attributes like cost, average rating, cuisine types, and popularity metrics.

## ML Models Used

* **K-Means Clustering:** A centroid-based algorithm, chosen as the final model for its balance of performance, interpretability through clear cluster centroids, and scalability. It was used to partition restaurants into distinct, homogeneous groups.
* **Hierarchical Clustering (Agglomerative):** Builds a hierarchy of clusters, useful for understanding relationships between clusters through a dendrogram.
* **Gaussian Mixture Models (GMM):** A probabilistic model that assumes data points are generated from a mixture of Gaussian distributions, providing soft cluster assignments.

## Evaluation & Business Impact

For this unsupervised learning project, the key evaluation metrics considered for positive business impact were the **Silhouette Score** and **Davies-Bouldin Index**.

* **Silhouette Score:** A high score indicates well-defined and separated clusters, enabling Zomato to clearly differentiate restaurant segments for targeted strategies.
* **Davies-Bouldin Index:** A low score confirms cluster compactness and separation, ensuring the identified segments are distinct and actionable for resource allocation and focused campaigns.

The interpretability of the clusters, derived by analyzing the feature profiles of each segment, was paramount. This ensured that the analytical insights could be directly translated into tangible business strategies for Zomato, such as personalized recommendations, tailored restaurant support programs, and strategic market expansion initiatives.

## Tools & Technologies

* **Python:** Programming Language
* [cite_start]**Pandas:** Data manipulation and analysis [cite: 1]
* [cite_start]**NumPy:** Computationally efficient numerical operations [cite: 1]
* [cite_start]**Matplotlib & Seaborn:** Visualization and behavior analysis [cite: 1]
* [cite_start]**Scikit-learn:** Model building, preprocessing, and dimensionality reduction [cite: 1]
* **NLTK:** Natural Language Toolkit for text preprocessing (stopwords, lemmatization, tokenization, POS tagging)
* **WordCloud:** For generating word visualizations
* **`contractions` library:** For expanding contractions in text
* **Yellowbrick:** For visualizing clustering evaluation metrics (e.g., Elbow Method)
* **VS Code:** Integrated Development Environment
* **Jupyter Notebooks:** For interactive development and documentation

## How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/YourProjectName.git](https://github.com/YourGitHubUsername/YourProjectName.git)
    cd YourProjectName
    ```
2.  **Set Up Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source ./.venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk yellowbrick contractions wordcloud
    ```
4.  **Download NLTK Data:**
    Open a Python interpreter or run the following in a Jupyter cell within your active virtual environment:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```
5.  **Place Data Files:** Ensure `Zomato Restaurant names and Metadata.csv` and `Zomato Restaurant reviews.csv` are in the project root directory.
6.  **Run Jupyter Notebook:**
    Open the `Sample_ML_Submission_Template.ipynb` notebook in VS Code or Jupyter Lab/Notebook and run all cells sequentially. The notebook is designed to be executable in one go without errors.

## Future Work

* **Advanced NLP:** Implement aspect-based sentiment analysis for more granular insights from reviews (e.g., distinguishing sentiment on "food quality" vs. "service speed").
* **Geospatial Analysis:** Incorporate precise location data (latitude/longitude) to analyze restaurant distribution, identify geographical clusters, and understand market saturation by area.
* **Time Series Forecasting:** Develop models to predict future review trends or changes in customer ratings, aiding proactive business strategies.
* **Deployment:** Build a simple web application (e.g., using Flask or FastAPI) to host the trained clustering model, allowing Zomato to classify new restaurant data in real-time.
* **Recommendation Engine:** Develop a sophisticated recommendation system that leverages the identified restaurant clusters to provide highly personalized suggestions to users.
* **Supervised Applications:** Utilize the derived cluster labels as powerful features for supervised machine learning tasks, such as predicting future restaurant success, identifying churn risk, or optimizing commission structures.

## Contribution

* **Individual Project**

## License

This project is licensed under the MIT License - see the LICENSE.md file for details (Optional: Create a `LICENSE.md` file in your repository with the MIT license text).