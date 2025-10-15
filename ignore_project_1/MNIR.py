"""
Multinomial Inverse Regression for Sentiment Analysis
Analyzes sentiment of Guardian immigration headlines
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# For sentiment analysis
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    print(
        "TextBlob not available. Installing it is recommended for better sentiment analysis."
    )
    print("Run: uv pip install textblob")
    TEXTBLOB_AVAILABLE = False


def get_sentiment_polarity(text):
    """
    Get sentiment polarity score using TextBlob
    Returns a score between -1 (negative) and 1 (positive)
    """
    if TEXTBLOB_AVAILABLE:
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    else:
        # Simple rule-based backup
        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "support",
            "welcome",
            "success",
            "benefit",
            "help",
            "improve",
            "better",
            "best",
        ]
        negative_words = [
            "bad",
            "terrible",
            "negative",
            "crisis",
            "threat",
            "danger",
            "problem",
            "concern",
            "fear",
            "illegal",
            "raid",
            "deport",
        ]

        text_lower = str(text).lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)


def categorize_sentiment(polarity):
    """Categorize sentiment into discrete classes"""
    if polarity < -0.3:
        return 0  # Negative
    elif polarity < 0.1:
        return 1  # Neutral
    else:
        return 2  # Positive


def perform_mnir_analysis(df, text_column="headline"):
    """
    Perform Multinomial Inverse Regression analysis

    Steps:
    1. Calculate sentiment scores for each headline
    2. Categorize into sentiment classes
    3. Use multinomial logistic regression to model P(words | sentiment)
    4. This can be inverted to predict sentiment from text
    """
    print("=" * 80)
    print("MULTINOMIAL INVERSE REGRESSION SENTIMENT ANALYSIS")
    print("=" * 80)

    # Step 1: Get sentiment scores
    print("\nStep 1: Calculating sentiment polarity scores...")
    df["sentiment_polarity"] = df[text_column].apply(get_sentiment_polarity)

    # Step 2: Categorize sentiment
    print("Step 2: Categorizing sentiment into classes...")
    df["sentiment_class"] = df["sentiment_polarity"].apply(categorize_sentiment)

    # Step 3: Create text features
    print("Step 3: Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=500, min_df=5, max_df=0.8, ngram_range=(1, 2), stop_words="english"
    )

    X = vectorizer.fit_transform(df[text_column].fillna(""))
    y = df["sentiment_class"]

    # Step 4: Train multinomial logistic regression
    print("Step 4: Training multinomial logistic regression...")
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42
    )
    clf.fit(X, y)

    # Step 5: Get refined predictions
    print("Step 5: Getting refined sentiment predictions...")
    df["sentiment_proba_negative"] = clf.predict_proba(X)[:, 0]
    df["sentiment_proba_neutral"] = clf.predict_proba(X)[:, 1]
    df["sentiment_proba_positive"] = clf.predict_proba(X)[:, 2]
    df["sentiment_class_predicted"] = clf.predict(X)

    # Calculate "positivity score" (0 to 1 scale)
    df["positivity_score"] = (
        0.0 * df["sentiment_proba_negative"]
        + 0.5 * df["sentiment_proba_neutral"]
        + 1.0 * df["sentiment_proba_positive"]
    )

    return df, vectorizer, clf


def print_analysis_summary(df):
    """Print summary statistics of the sentiment analysis"""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nTotal articles analyzed: {len(df)}")

    print("\n--- Sentiment Distribution ---")
    sentiment_counts = df["sentiment_class_predicted"].value_counts().sort_index()
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for class_id, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{sentiment_labels[class_id]}: {count} ({percentage:.1f}%)")

    print("\n--- Positivity Score Statistics ---")
    print(f"Mean positivity score: {df['positivity_score'].mean():.3f}")
    print(f"Median positivity score: {df['positivity_score'].median():.3f}")
    print(f"Std dev: {df['positivity_score'].std():.3f}")
    print(f"Min: {df['positivity_score'].min():.3f}")
    print(f"Max: {df['positivity_score'].max():.3f}")

    print("\n--- Most Positive Headlines (Top 10) ---")
    top_positive = df.nlargest(10, "positivity_score")[
        ["headline", "positivity_score", "date"]
    ]
    for idx, row in top_positive.iterrows():
        print(
            f"{row['positivity_score']:.3f} | {row['date'][:10]} | {row['headline'][:70]}..."
        )

    print("\n--- Most Negative Headlines (Top 10) ---")
    top_negative = df.nsmallest(10, "positivity_score")[
        ["headline", "positivity_score", "date"]
    ]
    for idx, row in top_negative.iterrows():
        print(
            f"{row['positivity_score']:.3f} | {row['date'][:10]} | {row['headline'][:70]}..."
        )


def analyze_top_features(clf, vectorizer):
    """Analyze which words are most associated with each sentiment"""
    print("\n" + "=" * 80)
    print("TOP FEATURES BY SENTIMENT CLASS")
    print("=" * 80)

    feature_names = vectorizer.get_feature_names_out()
    sentiment_labels = ["Negative", "Neutral", "Positive"]

    for i, label in enumerate(sentiment_labels):
        print(f"\n--- {label} Sentiment ---")
        coef = clf.coef_[i]
        top_indices = np.argsort(coef)[-15:][::-1]
        top_features = [(feature_names[idx], coef[idx]) for idx in top_indices]

        for feature, weight in top_features:
            print(f"  {feature:20s}: {weight:.3f}")


def save_results(df, output_file="sentiment_results.csv"):
    """Save results to CSV"""
    output_columns = [
        "id",
        "headline",
        "date",
        "url",
        "section",
        "sentiment_polarity",
        "sentiment_class_predicted",
        "positivity_score",
        "sentiment_proba_negative",
        "sentiment_proba_neutral",
        "sentiment_proba_positive",
    ]

    df[output_columns].to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main execution function"""
    # Load data
    print("Loading data...")
    data_file = "guardian_immigration_all_years.csv"
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} articles from {data_file}")

    # Perform MNIR analysis
    df_analyzed, vectorizer, clf = perform_mnir_analysis(df, text_column="headline")

    # Print summary
    print_analysis_summary(df_analyzed)

    # Analyze features
    analyze_top_features(clf, vectorizer)

    # Save results
    save_results(df_analyzed)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return df_analyzed


if __name__ == "__main__":
    df_results = main()
