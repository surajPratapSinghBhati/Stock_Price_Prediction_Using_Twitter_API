# TAKING THE INPUT FROM THE USER

import tweepy
import yfinance as yf
from textblob import TextBlob  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import datetime


def get_twitter_client():
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAFZExAEAAAAApLmb9tSg8E%2FFo5T8xEIq1vCaWYM%3D4WPqPblaLflpvtkJeqyKWOybKPd53QWZ34Zd8VSngv14UiNYl2"
    return tweepy.Client(bearer_token=bearer_token)


def fetch_tweets(client, query, max_results=10):
    tweets = []
    try:
        response = client.search_recent_tweets(   
            query=query,
            max_results=max_results,
            tweet_fields=["text", "created_at", "lang"]
        )
        if response.data:
            tweets = [tweet.text for tweet in response.data]
        else:
            print("No tweets found for the query.")
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets: {e}")
    return tweets


def analyze_sentiments(tweets):
    sentiment_scores = []
    for tweet in tweets:
        analysis = TextBlob(tweet)
        sentiment_scores.append(analysis.sentiment.polarity)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0


def fetch_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data['Change'] = data['Close'] - data['Open']
    return data[['Open', 'Close', 'High', 'Low', 'Volume', 'Change']]



def prepare_dataset(stock_data, sentiment_score):
    stock_data['Sentiment'] = sentiment_score
    stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)  # 1 if price increases, 0 otherwise
    stock_data = stock_data.dropna() 
    return stock_data


def train_and_predict(models, data):
    X = data[['Open', 'High', 'Low', 'Volume', 'Change', 'Sentiment']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    plt.figure(figsize=(10, 6))

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate accuracy for each model
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = accuracy

        # Visualization of Predictions vs Actual Labels
        plt.plot(y_test.values, label=f"Actual Labels", marker='o', linestyle='dashed')
        plt.plot(predictions, label=f"{model_name} Predictions", marker='x', linestyle='dashed')

    return results


if __name__ == "__main__":
    # Take user input
    stock_symbol = input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()
    query = input("Enter the search query for tweets (e.g., 'Apple stock'): ")
    days = int(input("Enter the number of days to analyze (e.g., 30): "))

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)

    # Initialize Twitter client
    client = get_twitter_client()

    # Fetch tweets and analyze sentiment
    tweets = fetch_tweets(client, query, max_results=50)
    sentiment_score = analyze_sentiments(tweets)

    # Fetch stock price data
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

    # Prepare dataset
    dataset = prepare_dataset(stock_data, sentiment_score)

    if not dataset.empty:
        # Define models to test, including Decision Tree
        models = {
            'Linear Regression': LinearRegression(),
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }

        # Train models and get results
        results = train_and_predict(models, dataset)

        # Show accuracy for all models
        print("\nModel Performance (Accuracy):")
        for model_name, score in results.items():
            print(f"{model_name}: {score}")

        # Show the best model (with the highest accuracy)
        best_model_name = max(results, key=results.get)
        print(f"\nBest Model: {best_model_name} with Accuracy: {results[best_model_name]}")

        # Display the plot
        plt.title("Model Predictions vs Actual Labels")
        plt.xlabel("Sample Index")
        plt.ylabel("Label (1 = Increase, 0 = Decrease)")
        plt.legend()
        plt.grid()
        plt.show()

    else:
        print("Insufficient data to train the model.")



