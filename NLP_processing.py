import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')


class TaskNLPPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Add common task-related stopwords
        additional_stopwords = {'task', 'need', 'needs', 'required', 'please', 'must', 'should'}
        self.stop_words.update(additional_stopwords)

    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_text(self, text):
        """Tokenize text into words"""
        if not text:
            return []
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]

    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens (alternative to stemming)"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(self, text, use_stemming=True):
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize_text(cleaned_text)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Apply stemming or lemmatization
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        else:
            tokens = self.lemmatize_tokens(tokens)

        return tokens

    def tokens_to_string(self, tokens):
        """Convert token list back to string"""
        return ' '.join(tokens)


def process_tasks_dataset(input_file='tasks_dataset.csv', output_file='task_preprocessed_data.csv'):
    """
    Main function to process the tasks dataset
    """
    print("Loading tasks dataset...")

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} tasks from {input_file}")

        # Display basic info about the dataset
        print("\nDataset columns:", df.columns.tolist())
        print("\nFirst few task descriptions:")
        print(df['task_description'].head())

    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Initialize preprocessor
    preprocessor = TaskNLPPreprocessor()

    print("\nProcessing task descriptions...")

    # Apply preprocessing to task descriptions
    df['task_description_cleaned'] = df['task_description'].apply(
        lambda x: preprocessor.clean_text(x)
    )

    df['task_description_tokens'] = df['task_description'].apply(
        lambda x: preprocessor.preprocess_text(x, use_stemming=True)
    )

    df['task_description_processed'] = df['task_description_tokens'].apply(
        lambda x: preprocessor.tokens_to_string(x)
    )

    # Create additional columns for analysis
    df['token_count'] = df['task_description_tokens'].apply(len)
    df['original_word_count'] = df['task_description'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    print("Preprocessing completed!")

    # Display some statistics
    print(f"\nPreprocessing Statistics:")
    print(f"Average original word count: {df['original_word_count'].mean():.2f}")
    print(f"Average processed token count: {df['token_count'].mean():.2f}")
    print(f"Reduction ratio: {(1 - df['token_count'].mean() / df['original_word_count'].mean()) * 100:.1f}%")

    # Show examples of preprocessing
    print("\nExample of preprocessing:")
    for i in range(min(3, len(df))):
        print(f"\nTask {i + 1}:")
        print(f"Original: {df.iloc[i]['task_description']}")
        print(f"Processed: {df.iloc[i]['task_description_processed']}")
        print(f"Tokens: {df.iloc[i]['task_description_tokens']}")

    # Select columns for output
    output_columns = [
        'taskid',
        'task_description',
        'task_description_processed',
        'task_description_tokens',
        'priority',
        'category',
        'assigned_to_employeeid',
        'token_count',
        'original_word_count'
    ]

    # Create output dataframe
    output_df = df[output_columns].copy()

    # Convert tokens list to string representation for CSV storage
    output_df['task_description_tokens'] = output_df['task_description_tokens'].apply(
        lambda x: '|'.join(x) if x else ''
    )

    # Save to CSV
    try:
        output_df.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return

    return output_df


def analyze_preprocessing_results(df):
    """
    Analyze the results of preprocessing
    """
    print("\n" + "=" * 50)
    print("PREPROCESSING ANALYSIS")
    print("=" * 50)

    # Category-wise analysis
    print("\nToken count by category:")
    category_stats = df.groupby('category')['token_count'].agg(['mean', 'std', 'min', 'max'])
    print(category_stats.round(2))

    # Priority-wise analysis
    print("\nToken count by priority:")
    priority_stats = df.groupby('priority')['token_count'].agg(['mean', 'std', 'min', 'max'])
    print(priority_stats.round(2))

    # Most common processed words
    print("\nMost common words after preprocessing:")
    all_tokens = []
    for tokens_str in df['task_description_tokens']:
        if tokens_str:
            all_tokens.extend(tokens_str.split('|'))

    from collections import Counter
    word_freq = Counter(all_tokens)
    print("Top 20 most common words:")
    for word, freq in word_freq.most_common(20):
        print(f"  {word}: {freq}")


if __name__ == "__main__":
    print("Starting NLP preprocessing for Tasks Dataset")
    print("=" * 50)

    # Process the dataset
    processed_df = process_tasks_dataset()

    if processed_df is not None:
        # Analyze results
        analyze_preprocessing_results(processed_df)

        print(f"\n{'=' * 50}")
        print("Processing completed successfully!")
        print("Files created:")
        print("- task_preprocessed_data.csv (main output)")
        print("\nThe processed dataset includes:")
        print("- Original task descriptions")
        print("- Cleaned and processed text")
        print("- Individual tokens (pipe-separated)")
        print("- Token counts and statistics")