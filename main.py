#importing libraries

#pip install nltk
#pip install twython
#pip install textblob
#pip install wordcloud


from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
#nltk.download('wordnet')
#nltk.download('omw-1.4')
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

#adjusting row column settings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#loading the dataset
df = pd.read_csv("D:/Sentiment Analysis Project/tweets_labeled.csv")
df.head()

df.columns
df.shape
df.info()

# Converting 'date' column to datetime
df["date"] = pd.to_datetime(df["date"])

# Checking if the date is already tz-aware, if not, localize to UTC
if df["date"].dt.tz is None:
    df["date"] = df["date"].dt.tz_localize("UTC")

# Converting the timezone to 'Europe/Istanbul'
df["date"] = df["date"].dt.tz_convert("Europe/Istanbul")

# Extracting the month name
df['month'] = df['date'].dt.month_name()

# Converting tweet text to lowercase
df["tweet"] = df["tweet"].str.lower()


df

#creating the variable seasons
seasons = {'January': 'Winter',
           'February': 'Winter',
           'March': 'Spring',
           'April': 'Spring',
           'May': 'Spring',
           'June': 'Summer',
           'July': 'Summer',
           'August': 'Summer',
           'September': 'Autumn',
           'October': 'Autumn',
           'November': 'Autumn',
           'December': 'Winter'}

df["seasons"] = df["month"].map(seasons)
df.head()

#creation of day variable
df["days"] = [date.strftime('%A') for date in df["date"]]
df["hour"] = df["date"].dt.hour

df.head()
df

df['4hour_interval'] = (df['hour'] // 2) * 2

interval = {0: '0-2',
            2: '2-4',
            4: '4-6',
            6: '6-8',
            8: '8-10',
            10: '10-12',
            12: '12-14',
            14: '14-16',
            16: '16-18',
            18: '18-20',
            20: '20-22',
            22: '22-24'
            }
df['4hour_interval'] = df['4hour_interval'].map(interval)
df.head()
df

df["time_interval"] = df["4hour_interval"].replace({"0-2": "22-02",
                                                   "22-24": "22-02",
                                                   "2-4": "02-06",
                                                   "4-6": "02-06",
                                                   "6-8": "06-10",
                                                   "8-10": "06-10",
                                                   "10-12": "10-14",
                                                   "12-14": "10-14",
                                                   "14-16": "14-18",
                                                   "16-18": "14-18",
                                                   "18-20": "18-22",
                                                   "20-22": "18-22"})


df.head()

df.drop(["4hour_interval", "hour"], axis=1, inplace=True)

label = {1: 'positive',
        -1: 'negative',
        0: 'neutral'
         }

df['label'] = df['label'].map(label)



def summary(df, col_name, plot=False, save_plots=False):
    """
    Generate summary statistics and optional plots for a specified column in negative tweets.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing text data.
    col_name (str): The column to analyze.
    plot (bool): If True, plot the column distribution.
    save_plots (bool): If True, save the plot as a file.
    
    Returns:
    None: This function only prints summary statistics and plots the results.
    """
    # Filter negative tweets
    df_filtered = df[df["label"] == 'negative']
    
    # If no data after filtering, print a message
    if df_filtered.empty:
        print(f"No negative tweets found in the data for column: {col_name}")
        return

    # Check if the column exists in the DataFrame
    if col_name not in df.columns:
        print(f"Column '{col_name}' not found in the DataFrame.")
        return

    # Calculate value counts and ratio
    count_series = df_filtered[col_name].value_counts()  # Get counts
    ratio_series = df_filtered[col_name].value_counts(normalize=True) * 100  # Get ratio
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({col_name: count_series, 'Ratio (%)': ratio_series})
    
    # Print summary
    print(f"Summary for column: {col_name}")
    print(summary_df)
    print("---------------------------------------------")

    # Plot distribution if plot=True
    if plot:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=col_name, data=df_filtered, palette="magma")
        ax.set_title(f"{col_name.capitalize()} Distribution in Negative Tweets")
        ax.set_xlabel(col_name.capitalize())
        ax.set_ylabel("Count")

        # Annotate percentages
        total_count = len(df_filtered)
        for p in ax.patches:
            height = p.get_height()
            percentage = f'{(height / total_count) * 100:.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, height), 
                        ha='center', va='center', fontsize=10, color='black', 
                        xytext=(0, 5), textcoords='offset points')

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot if save_plots=True
        if save_plots:
            plt.savefig(f"{col_name}_distribution.png")

        # Show plot
        plt.show()

# Example usage with a list of columns
cols = ["time_interval", "days", "seasons"]
for col in cols:
    summary(df, col, plot=True, save_plots=True)
    
    

###### Text preprocessing ############################
#Lowercases, punctuation, numbers and newline characters

def clean_text(text):
    """
    Clean and preprocess text data.

    This function performs several cleaning operations on text data:
    - Lowercases the text (Case Folding)
    - Removes punctuation
    - Removes numbers
    - Removes newline characters

    Parameters:
    text (pandas.Series): A pandas Series containing text data.

    Returns:
    pandas.Series: A pandas Series with cleaned text.
    """
    # Lowercasing (Case Folding)
    text = text.str.lower()
    # Removing punctuations, numbers, and newline characters
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    return text

df["tweet"] = clean_text(df["tweet"])
df["tweet"]


#nltk.download("stopwords")

stop_words = stopwords.words("turkish")

# Stopwords
def remove_stopwords(text):
    """
    Remove stopwords from text data.

    This function filters out common stopwords from the text data. 
    Stopwords are removed based on the NLTK's English stopwords list.

    Parameters:
    text (pandas.Series): A pandas Series containing text data.

    Returns:
    pandas.Series: A pandas Series with stopwords removed from the text.
    """
    # Removing stopwords
    text = text.apply(lambda x: " ".join(word for word in str(x).split() if word not in stop_words))
    return text

df["tweet"] = remove_stopwords(df["tweet"])
df["tweet"]

# Rare Words and Frequent Words
def remove_rare_words(df, column_name, n_rare_words=1000):
    """
    Remove rare words from a specified column in a pandas DataFrame.

    This function identifies and removes the least frequently occurring words
    in the text data. It is useful for removing rare words that might not contribute
    significantly to the analysis or modeling.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column in the DataFrame to clean.
    n_rare_words (int): The number of least frequent words to remove.

    Returns:
    pandas.DataFrame: A DataFrame with rare words removed from the specified column.
    """
    # Identifying the rare words
    freq = pd.Series(' '.join(df[column_name]).split()).value_counts()
    rare_words = freq[-n_rare_words:]

    # Removing the rare words
    df[column_name] = df[column_name].apply(lambda x: " ".join(word for word in x.split() if word not in rare_words))
    return df

df = remove_rare_words(df, 'tweet', 1000)
df["tweet"]


#nltk.download('punkt')

# Tokenization
df["tweet"].apply(lambda x: TextBlob(x).words)

# Lemmatization

def apply_lemmatization(df, column_name):
    """
    Apply lemmatization to a specified column in a pandas DataFrame.

    This function performs lemmatization on the text data in the specified column.
    Lemmatization involves reducing each word to its base or root form.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column in the DataFrame to process.

    Returns:
    pandas.DataFrame: A DataFrame with lemmatized text in the specified column.
    """
    # Applying lemmatization
    df[column_name] = df[column_name].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df
df = apply_lemmatization(df, 'tweet')
df["tweet"]

# Data Visualization
'''
Barplot: Barplot is a type of graph used to visualise categorical data. It is often used to show frequencies 
or relationships of frequently occurring categorical values.

Word Cloud: Word Cloud is a type of chart used to visualise text data and highlight the importance of certain words.
They are visually represented in different sizes and colours according to the frequency of the words in the text.
'''
# 1. Term Frequency Calculation and Bar Chart
tf = df["tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# 2. Word Cloud Generation
text = " ".join(i for i in df.tweet)
wordcloud = WordCloud(max_font_size=100, max_words= 1000, background_color= "black").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Word Cloud")
plt.axis("off")
plt.show()


#Sentiment Analysis

df["label"] = LabelEncoder().fit_transform(df["label"])
df.head()

df.dropna(axis=0, inplace=True)

#TF-IDF Word Level
'''TF-IDF Word Level is a measure used to determine the importance of a term within a document, and this measure 
is based on the frequency of the term within the document and its prevalence across all documents.
'''
tf_idfVectorizer = TfidfVectorizer()

X = tf_idfVectorizer.fit_transform(df["tweet"])
y = df["label"]

# Define the mapping between numerical labels and their original string values
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Adjust if your LabelEncoder used a different order

# Create the 'sentiment' column
df['sentiment'] = df['label'].map(label_mapping)

# Verify
print(df[['label', 'sentiment']].head())
# Heatmap(Exploring interactions between time of day and season.)
heatmap_data = pd.crosstab(df['time_interval'], df['seasons'], values=df['sentiment'], aggfunc='count')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Tweet Volume by Time Interval and Season')
plt.xlabel('Season')
plt.ylabel('Time Interval')
plt.show()

#Comparing sentiment distribution across seasons.
seasonal_sentiment = df.groupby(['seasons', 'sentiment']).size().unstack()
seasonal_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution by Season')
plt.xlabel('Season')
plt.xticks(rotation=45)
plt.show()

#################### Model Building ########################

'''from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd

# Define models and their parameter grids for tuning
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=10000),
        "params": {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'saga']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    "Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            'alpha': [0.1, 1, 10]
        }
    }
}

# Store results
cv_scores = {}
best_scores = {}
best_params = {}

# Perform cross-validation for baseline performance
for model_name, model_config in models.items():
    print(f"\n=== Cross-Validation for {model_name} ===")
    cv_score = cross_val_score(model_config["model"], X, y, cv=10, scoring='accuracy', n_jobs=-1)
    mean_cv_score = cv_score.mean()
    cv_scores[model_name] = mean_cv_score
    print(f"Mean CV Accuracy: {mean_cv_score:.4f}")

# Hyperparameter tuning using GridSearchCV
for model_name, model_config in models.items():
    print(f"\n=== Tuning {model_name} ===")
    grid_search = GridSearchCV(
        estimator=model_config["model"],
        param_grid=model_config["params"],
        cv=10,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    best_scores[model_name] = grid_search.best_score_
    best_params[model_name] = grid_search.best_params_
    
    print(f"Best Accuracy: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

# Compare models
results_df = pd.DataFrame({
    'Model': best_scores.keys(),
    'Baseline CV Accuracy': cv_scores.values(),
    'Tuned Accuracy': best_scores.values(),
    'Best Parameters': best_params.values()
}).sort_values(by='Tuned Accuracy', ascending=False)

print("\n=== Final Model Comparison ===")
print(results_df)

# Identify and print the best model
best_model_name = results_df.iloc[0]['Model']
best_model_accuracy = results_df.iloc[0]['Tuned Accuracy']
best_model_params = results_df.iloc[0]['Best Parameters']

print(f"\nBest Model: {best_model_name}")
print(f"Tuned Accuracy: {best_model_accuracy:.4f}")
print(f"Best Parameters: {best_model_params}")
'''

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize SVM model with best parameters
svm_model = SVC(C=1, kernel='rbf', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Evaluate on test data
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Perform cross-validation
cv_scores = cross_val_score(svm_model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
print(f"\nCross-Validation Mean Accuracy: {cv_scores.mean():.4f}")

'''
Test Accuracy: 69.06% – The model correctly predicts labels for approximately 69% of the test data.

Cross-Validation Accuracy: 66.97% – This suggests the model performs consistently across different data splits.

Classification Report:
Class 1 (Majority Class) is well predicted with 98% recall, meaning almost all actual Class 1 instances are correctly classified.

Class 0 & Class 2 suffer from low recall (13% and 17%, respectively), meaning many instances of these classes are misclassified.

Precision is higher than recall for Class 0 & Class 2, indicating the model is more confident when it does predict these classes but often fails to identify them.

Confusion Matrix Insights:
Class 1 is dominant: Most of its predictions are correct (1657 out of 1690).

Class 0 & Class 2 are often misclassified as Class 1, leading to an imbalance issue in prediction.

Key Takeaways & Next Steps:
The model is biased towards Class 1, likely due to an imbalance in the dataset.

Improving Class 0 & Class 2 predictions could involve:

Handling class imbalance (e.g., oversampling/undersampling).

Using different metrics (e.g., F1-score, balanced accuracy).

Trying other kernels or feature engineering to improve separability.

'''

# Twitter 2021 data

df_tweet_21 = pd.read_csv(r"D:\Sentiment Analysis Project\tweets_21.csv")
df_tweet_21.head()

# Feature Engineering
# Lowercases, Punctuation, Numbers and Newline Characters

def clean_text(text):
    """
    Clean and preprocess text data.

    This function performs several cleaning operations on text data:
    - Lowercases the text (Case Folding)
    - Removes punctuation
    - Removes numbers
    - Removes newline characters

    Parameters:
    text (pandas.Series): A pandas Series containing text data.

    Returns:
    pandas.Series: A pandas Series with cleaned text.
    """
    # Lowercasing (Case Folding)
    text = text.str.lower()
    # Removing punctuations, numbers, and newline characters
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    return text
df_tweet_21["tweet"] = clean_text(df_tweet_21["tweet"])
df_tweet_21["tweet"]


# Stopwords

def remove_stopwords(text):
    """
    Remove stopwords from text data.

    This function filters out common stopwords from the text data. 
    Stopwords are removed based on the NLTK's English stopwords list.

    Parameters:
    text (pandas.Series): A pandas Series containing text data.

    Returns:
    pandas.Series: A pandas Series with stopwords removed from the text.
    """
    # Removing stopwords
    text = text.apply(lambda x: " ".join(word for word in str(x).split() if word not in stop_words))
    return text
df_tweet_21["tweet"] = remove_stopwords(df_tweet_21["tweet"])
df_tweet_21["tweet"]


# Rare Words and Frequent Words

def remove_rare_words(df, column_name, n_rare_words=1000):
    """
    Remove rare words from a specified column in a pandas DataFrame.

    This function identifies and removes the least frequently occurring words
    in the text data. It is useful for removing rare words that might not contribute
    significantly to the analysis or modeling.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column in the DataFrame to clean.
    n_rare_words (int): The number of least frequent words to remove.

    Returns:
    pandas.DataFrame: A DataFrame with rare words removed from the specified column.
    """
    # Identifying the rare words
    freq = pd.Series(' '.join(df[column_name]).split()).value_counts()
    rare_words = freq[-n_rare_words:]

    # Removing the rare words
    df[column_name] = df[column_name].apply(lambda x: " ".join(word for word in x.split() if word not in rare_words))
    return df
df_tweet_21 = remove_rare_words(df_tweet_21, 'tweet', 1000)
df_tweet_21["tweet"]


# Tokenization

df_tweet_21["tweet"].apply(lambda x: TextBlob(x).words)

# Lemmatization

def apply_lemmatization(df, column_name):
    """
    Apply lemmatization to a specified column in a pandas DataFrame.

    This function performs lemmatization on the text data in the specified column.
    Lemmatization involves reducing each word to its base or root form.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column in the DataFrame to process.

    Returns:
    pandas.DataFrame: A DataFrame with lemmatized text in the specified column.
    """
    # Applying lemmatization
    df[column_name] = df[column_name].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    return df
df_tweet_21 = apply_lemmatization(df_tweet_21, 'tweet')
df_tweet_21["tweet"]


# 1. Term Frequency Calculation and Bar Chart
tf = df_tweet_21["tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()
    
# 2. Word Cloud Generation
text = " ".join(i for i in df_tweet_21.tweet)
wordcloud = WordCloud(max_font_size=100, max_words= 1000, background_color= "black").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Word Cloud")
plt.axis("off")
plt.show()

# Prediction
tweet_tfidf = tf_idfVectorizer.transform(df_tweet_21["tweet"])
predictions = svm_model.predict(tweet_tfidf)
df_tweet_21["label"] = predictions

df_tweet_21.head()





