import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import re
import string
from nltk.corpus import stopwords

# Set of stopwords
STOPWORDS = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

##Tags the words in the tweets
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:          
        return(None)

##Lemmatizes the words in tweets and returns the cleaned and lemmatized tweet
def lemmatize_tweet(tweet):
    #tokenize the tweet and find the POS tag for each token
    tweet = tweet_cleaner(tweet) #tweet_cleaner() will be the function you will write
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_tweet.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return(" ".join(lemmatized_tweet))

def tweet_cleaner(tweet):
    # Remove links (http/https)
    tweet = re.sub(r'http\S+|www\S+', '', tweet)

    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))

    # Remove emojis and non-ASCII characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)  # Remove non-ASCII chars

    # Remove non-alphanumeric tokens and words shorter than 3 characters
    tweet = " ".join([word for word in tweet.split() if word.isalnum() and len(word) > 2])

    # Remove stopwords
    tweet = " ".join([word for word in tweet.split() if word.lower() not in STOPWORDS])

    # Remove retweet markers and noise terms -> found from part b)
    tweet = re.sub(r'\b(rt|brt|amp|htt)\b', '', tweet, flags=re.IGNORECASE)

    return tweet