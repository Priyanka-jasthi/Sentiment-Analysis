
#importing libraries

#pip install nltk
#pip install twython
#pip install textblob
#pip install wordcloud


from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import seaborn as sns
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
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

#processing date and time data
df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].dt.tz_localize("UTC")
df["date"] = df["date"].dt.tz_convert("Europe/Istanbul")
#extracting month name
df['month'] = df['date'].dt.month_name()
#converting tweet text to lowercase
df["tweet"] = df["tweet"].str.lower()

df.info()

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
        ax = sns.countplot(x=col_name, data=df_filtered, palette="viridis")
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


nltk.download("stopwords")

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

#Calculation of Term Frequencies & Barplot & Word Cloud

def plot_tf_and_wordcloud(df, column_name, tf_threshold=2000, max_font_size=50, max_words=100, background_color="black"):
    """
    Calculate term frequency (TF) and generate a word cloud for a specified column in a pandas DataFrame.

    This function performs two main tasks:
    1. Term Frequency Calculation and Bar Chart: Calculates the frequency of each word in the specified column and plots a bar chart for words with a frequency above a certain threshold.
    2. Word Cloud Generation: Generates and displays a word cloud based on the text in the specified column.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column to analyze.
    tf_threshold (int): The threshold for term frequency to be included in the bar chart.
    max_font_size (int): Maximum font size for the word cloud.
    max_words (int): The maximum number of words for the word cloud.
    background_color (str): Background color for the word cloud.

    Returns:
    None: This function only plots the results and does not return any value.
    """
    # 1. Term Frequency Calculation and Bar Chart
    tf = df[column_name].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    high_tf = tf[tf["tf"] > tf_threshold]
    
    plt.figure(figsize=(12, 6))
    ax = high_tf.plot.bar(x="words", y="tf", title="Term Frequency Bar Chart", legend=False, color='skyblue')
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    
    # Add value labels on each bar
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.xticks(rotation=45)
    plt.show()

    # 2. Word Cloud Generation
    text = " ".join(i for i in df[column_name])
    wordcloud = WordCloud(max_font_size=max_font_size, max_words=max_words, background_color=background_color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Word Cloud")
    plt.axis("off")
    plt.show()
plot_tf_and_wordcloud(df, 'tweet')


#Sentiment Analysis
df["label"] = LabelEncoder().fit_transform(df["label"])
df.head()

df.dropna(axis=0, inplace=True)

#TF-IDF Word Level

tf_idfVectorizer = TfidfVectorizer()

X = tf_idfVectorizer.fit_transform(df["tweet"])
y = df["label"]

#Modelling
############## Logistic Regression #################

log_model = LogisticRegression(max_iter=10000).fit(X, y)

# Cross Validation
cross_val_score(log_model,
                X,
                y,
                scoring="accuracy",
                cv=10).mean()



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



def plot_tf_and_wordcloud(df, column_name, tf_threshold=2000, max_font_size=50, max_words=100, background_color="black"):
    """
    Calculate term frequency (TF) and generate a word cloud for a specified column in a pandas DataFrame.

    This function performs two main tasks:
    1. Term Frequency Calculation and Bar Chart: Calculates the frequency of each word in the specified column and plots a bar chart for words with a frequency above a certain threshold.
    2. Word Cloud Generation: Generates and displays a word cloud based on the text in the specified column.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame containing the text data.
    column_name (str): The name of the column to analyze.
    tf_threshold (int): The threshold for term frequency to be included in the bar chart.
    max_font_size (int): Maximum font size for the word cloud.
    max_words (int): The maximum number of words for the word cloud.
    background_color (str): Background color for the word cloud.

    Returns:
    None: This function only plots the results and does not return any value.
    """
    # 1. Term Frequency Calculation and Bar Chart
    tf = df[column_name].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    high_tf = tf[tf["tf"] > tf_threshold]
    
    plt.figure(figsize=(12, 6))
    ax = high_tf.plot.bar(x="words", y="tf", title="Term Frequency Bar Chart", legend=False, color='skyblue')
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    
    # Add value labels on each bar
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.xticks(rotation=45)
    plt.show()

    # 2. Word Cloud Generation
    text = " ".join(i for i in df[column_name])
    wordcloud = WordCloud(max_font_size=max_font_size, max_words=max_words, background_color=background_color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Word Cloud")
    plt.axis("off")
    plt.show()
plot_tf_and_wordcloud(df_tweet_21, 'tweet')

# Prediction

tweet_tfidf = tf_idfVectorizer.transform(df_tweet_21["tweet"])
predictions = log_model.predict(tweet_tfidf)
df_tweet_21["label"] = predictions

df_tweet_21.head()
