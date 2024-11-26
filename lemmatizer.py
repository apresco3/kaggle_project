import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import re

lemmatizer = WordNetLemmatizer()

def nltk_tag_to_wordnet_tag(nltk_tag):
    """Convert NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_tweet(tweet):
    """Lemmatize the tweet."""
    # Clean the tweet first
    
    tweet = tweet_cleaner(tweet)
    nltk_tagged = nltk.pos_tag(word_tokenize(tweet))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_tweet.append(word)
        else:
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_tweet)


def tweet_cleaner(tweet):
    """Clean the tweet by removing mentions, links, emojis, RT patterns, and other noise."""
    # Decode byte strings if needed
    if isinstance(tweet, bytes):
        tweet = tweet.decode('utf-8')

    # Compile patterns once for efficiency
    emoji_pattern = re.compile("["  # Emojis and symbols
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    url_pattern = re.compile(r'http\S+|www\S+')
    mention_pattern = re.compile(r'@\w+')
    rt_pattern = re.compile(r"(^RT\s*:|^b' RT\s*:|^b')")  # Matches "RT :", "b' RT :", or "b'"

    # Perform cleaning
    tweet = rt_pattern.sub('', tweet)       # Remove RT patterns
    tweet = mention_pattern.sub('', tweet)  # Remove mentions
    tweet = url_pattern.sub('', tweet)      # Remove URLs
    tweet = emoji_pattern.sub('', tweet)    # Remove emojis

    # Strip leading/trailing whitespace
    return tweet.strip()
