import pandas as pd
import spacy
import re
import nltk
from nltk.stem import WordNetLemmatizer
import ssl
import certifi


# Check if SSL certificates are available
if not hasattr(ssl, "_create_default_https_context"):
    # If SSL certificates are missing, add the certifi CA bundle
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = ssl._create_unverified_context

# Set the NLTK data path to use Certifi's certificate bundle
nltk.data.path.append("/path/to/nltk_data")
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('wordnet')

file_path = "./reviews_Musical_Instruments_5.json"

df = pd.read_json(file_path, lines=True)

df = df[df['overall'] != 3]

# Convert all text to lowercase to maintain uniformity
df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lambda x: x.lower())

def remove_emojis(text):
    # Define regex pattern to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    # Remove emojis from text
    return emoji_pattern.sub(r'', text)

df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lambda x: remove_emojis(x))


# ii. Removing Punctuation and Special Characters, replace hyphen and parentheses with space
# Remove punctuation (excluding hyphens) and special characters using regular expressions
# Add a space in place of punctuation to prevent word concatenation
df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lambda x: re.sub(r'[^\w\s-]', ' ', x))

# Replace hyphens and parentheses with space
df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lambda x: x.replace('-', ' ').replace('(', ' ').replace(')', ' '))

# Collapse multiple spaces into a single space
df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lambda x: re.sub(' +', ' ', x))

# Replace the pattern with an empty string to remove it
df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lambda x: re.sub(r"\d", '', x))

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# Function to preprocess text using spaCy
def preprocess_text(text):
    doc = nlp(text)
    # Remove stopwords and lemmatize
    filtered_words = [token.lemma_ for token in doc if not token.is_stop]
    # Join the filtered words to form a clean text
    clean_text = ' '.join(filtered_words)
    return clean_text

# Apply the preprocessing function to selected columns
columns_to_process = ['reviewText']
for col in columns_to_process:
    df[col] = df[col].apply(preprocess_text)


lemmatizer = WordNetLemmatizer()
count=-1
# 
# Define a function to perform lemmatization
def lemmatize_text(text):
    # Tokenize the text and apply lemmatization to each word
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in re.findall(r'\w+', text)])
    return lemmatized_text

# Apply the lemmatization function to the selected columns
df[['reviewText', 'summary']] = df[['reviewText', 'summary']].applymap(lemmatize_text)


df = df.dropna(subset=['reviewText'])
df["length_review"] = df["reviewText"].apply(lambda x: len(x.split()))


df = df.dropna(subset=['summary'])
df["length_summary"] = df["summary"].apply(lambda x: len(x.split()))

# Specify the path where you want to save the CSV file
csv_file_path = "./processedData.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)