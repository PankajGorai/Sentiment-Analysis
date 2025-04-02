import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Download necessary NLTK resources
nltk.download("punkt")

# List to store results
sentiments = []

# Loop for user input
while True:
    sentence = input("Enter a sentence (or type 'exit' to quit): ")
    if sentence.lower() == "exit":
        break

    # Perform sentiment analysis
    analysis = TextBlob(sentence)
    polarity = analysis.sentiment.polarity

    # Determine sentiment category
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Store result
    sentiments.append(sentiment)
    print(f"Sentiment: {sentiment}\n")

# Convert to DataFrame for visualization
df = pd.DataFrame(sentiments, columns=["Predicted_Sentiment"])

# Plot results
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Predicted_Sentiment"], palette="coolwarm")
plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
