import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from typing import List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_nltk_resources() -> None:
    """Initialize required NLTK resources."""
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]

    for resource_path, download_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading {download_name}...")
            nltk.download(download_name, quiet=True)


class TextPreprocessor:
    """Text preprocessing utility for task description analysis."""

    def __init__(self, custom_stopwords: Optional[set] = None):
        """
        Initialize the text preprocessor.

        Args:
            custom_stopwords: Additional stopwords to include in filtering
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Initialize base stopwords
        self.stopwords_set = set(stopwords.words('english'))

        # Add domain-specific stopwords
        default_custom_stopwords = {
            'task', 'need', 'needs', 'required', 'please', 'must', 'should'
        }

        if custom_stopwords:
            default_custom_stopwords.update(custom_stopwords)

        self.stopwords_set.update(default_custom_stopwords)

    def clean_text(self, text: Union[str, None]) -> str:
        """
        Perform basic text cleaning operations.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ""

        # Convert to lowercase and ensure string type
        cleaned_text = str(text).lower()

        # Remove non-alphabetic characters, preserve spaces
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)

        # Normalize whitespace
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []
        return word_tokenize(text)

    def filter_stopwords(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """
        Remove stopwords and short tokens.

        Args:
            tokens: List of tokens to filter
            min_length: Minimum token length to retain

        Returns:
            Filtered list of tokens
        """
        return [
            token for token in tokens
            if token not in self.stopwords_set and len(token) > min_length
        ]

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to tokens.

        Args:
            tokens: List of tokens to stem

        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.

        Args:
            tokens: List of tokens to lemmatize

        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def process_text(self, text: Union[str, None], use_stemming: bool = True) -> List[str]:
        """
        Execute complete text preprocessing pipeline.

        Args:
            text: Input text to process
            use_stemming: Whether to use stemming (True) or lemmatization (False)

        Returns:
            List of processed tokens
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)

        # Step 2: Tokenize
        tokens = self.tokenize(cleaned_text)

        # Step 3: Filter stopwords
        filtered_tokens = self.filter_stopwords(tokens)

        # Step 4: Apply normalization
        if use_stemming:
            processed_tokens = self.apply_stemming(filtered_tokens)
        else:
            processed_tokens = self.apply_lemmatization(filtered_tokens)

        return processed_tokens

    @staticmethod
    def tokens_to_text(tokens: List[str]) -> str:
        """
        Convert token list back to space-separated text.

        Args:
            tokens: List of tokens

        Returns:
            Space-separated text string
        """
        return ' '.join(tokens)


class TaskDatasetProcessor:
    """Main processor for task dataset NLP operations."""

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize the dataset processor.

        Args:
            preprocessor: Text preprocessor instance to use
        """
        self.preprocessor = preprocessor or TextPreprocessor()

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load task dataset from CSV file.

        Args:
            file_path: Path to input CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: For other loading errors
        """
        try:
            dataframe = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(dataframe)} records from {file_path}")
            logger.info(f"Dataset columns: {list(dataframe.columns)}")
            return dataframe
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as error:
            logger.error(f"Error loading dataset: {error}")
            raise

    def process_descriptions(self, dataframe: pd.DataFrame,
                             text_column: str = 'task_description',
                             use_stemming: bool = True) -> pd.DataFrame:
        """
        Process task descriptions in the dataset.

        Args:
            dataframe: Input DataFrame
            text_column: Name of column containing text to process
            use_stemming: Whether to use stemming vs lemmatization

        Returns:
            DataFrame with additional processed text columns
        """
        processed_df = dataframe.copy()

        # Generate processed text columns
        processed_df[f'{text_column}_cleaned'] = processed_df[text_column].apply(
            self.preprocessor.clean_text
        )

        processed_df[f'{text_column}_tokens'] = processed_df[text_column].apply(
            lambda x: self.preprocessor.process_text(x, use_stemming)
        )

        processed_df[f'{text_column}_processed'] = processed_df[f'{text_column}_tokens'].apply(
            self.preprocessor.tokens_to_text
        )

        # Calculate metrics
        processed_df['processed_token_count'] = processed_df[f'{text_column}_tokens'].apply(len)
        processed_df['original_word_count'] = processed_df[text_column].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )

        return processed_df

    def calculate_statistics(self, dataframe: pd.DataFrame) -> dict:
        """
        Calculate processing statistics.

        Args:
            dataframe: Processed DataFrame

        Returns:
            Dictionary containing statistics
        """
        stats = {
            'avg_original_words': dataframe['original_word_count'].mean(),
            'avg_processed_tokens': dataframe['processed_token_count'].mean(),
            'total_records': len(dataframe)
        }

        if stats['avg_original_words'] > 0:
            stats['reduction_percentage'] = (
                                                    1 - stats['avg_processed_tokens'] / stats['avg_original_words']
                                            ) * 100
        else:
            stats['reduction_percentage'] = 0

        return stats

    def save_processed_data(self, dataframe: pd.DataFrame, output_path: str,
                            columns_to_save: Optional[List[str]] = None) -> None:
        """
        Save processed dataset to CSV file.

        Args:
            dataframe: Processed DataFrame to save
            output_path: Output file path
            columns_to_save: Specific columns to include in output
        """
        if columns_to_save is None:
            columns_to_save = [
                'taskid', 'task_description', 'task_description_processed',
                'task_description_tokens', 'priority', 'category',
                'assigned_to_employeeid', 'processed_token_count', 'original_word_count'
            ]

        # Filter to available columns
        available_columns = [col for col in columns_to_save if col in dataframe.columns]
        output_df = dataframe[available_columns].copy()

        # Convert token lists to pipe-separated strings for CSV storage
        token_columns = [col for col in output_df.columns if 'tokens' in col]
        for col in token_columns:
            output_df[col] = output_df[col].apply(
                lambda x: '|'.join(x) if isinstance(x, list) and x else ''
            )

        try:
            output_df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as error:
            logger.error(f"Error saving file: {error}")
            raise


class AnalysisReporter:
    """Generate analysis reports for processed text data."""

    @staticmethod
    def generate_category_analysis(dataframe: pd.DataFrame,
                                   group_column: str = 'category',
                                   metric_column: str = 'processed_token_count') -> pd.DataFrame:
        """
        Generate statistics grouped by category.

        Args:
            dataframe: Input DataFrame
            group_column: Column to group by
            metric_column: Metric column to analyze

        Returns:
            DataFrame with grouped statistics
        """
        if group_column not in dataframe.columns:
            logger.warning(f"Column '{group_column}' not found in dataset")
            return pd.DataFrame()

        return dataframe.groupby(group_column)[metric_column].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)

    @staticmethod
    def get_token_frequency(dataframe: pd.DataFrame,
                            token_column: str = 'task_description_tokens',
                            top_n: int = 20) -> List[tuple]:
        """
        Calculate token frequency across all processed texts.

        Args:
            dataframe: Input DataFrame
            token_column: Column containing token lists
            top_n: Number of top tokens to return

        Returns:
            List of (token, frequency) tuples
        """
        all_tokens = []

        for tokens in dataframe[token_column]:
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
            elif isinstance(tokens, str) and tokens:
                # Handle pipe-separated token strings
                all_tokens.extend(tokens.split('|'))

        token_counter = Counter(all_tokens)
        return token_counter.most_common(top_n)

    def print_analysis_report(self, dataframe: pd.DataFrame,
                              statistics: dict) -> None:
        """
        Print comprehensive analysis report.

        Args:
            dataframe: Processed DataFrame
            statistics: Processing statistics dictionary
        """
        print("\n" + "=" * 60)
        print("TEXT PROCESSING ANALYSIS REPORT")
        print("=" * 60)

        # Overall statistics
        print(f"\nProcessing Summary:")
        print(f"Total records processed: {statistics['total_records']:,}")
        print(f"Average original word count: {statistics['avg_original_words']:.2f}")
        print(f"Average processed token count: {statistics['avg_processed_tokens']:.2f}")
        print(f"Text reduction: {statistics['reduction_percentage']:.1f}%")

        # Category analysis
        if 'category' in dataframe.columns:
            print(f"\nToken Count Analysis by Category:")
            category_stats = self.generate_category_analysis(dataframe, 'category')
            if not category_stats.empty:
                print(category_stats)

        # Priority analysis
        if 'priority' in dataframe.columns:
            print(f"\nToken Count Analysis by Priority:")
            priority_stats = self.generate_category_analysis(dataframe, 'priority')
            if not priority_stats.empty:
                print(priority_stats)

        # Token frequency
        print(f"\nMost Frequent Tokens:")
        frequent_tokens = self.get_token_frequency(dataframe)
        for token, frequency in frequent_tokens:
            print(f"  {token}: {frequency}")


def main(input_file_path: str = 'datasets/tasks_dataset.csv',
         output_file_path: str = 'datasets/task_preprocessed_data.csv') -> None:
    """
    Main execution function for task dataset processing.

    Args:
        input_file_path: Path to input CSV file
        output_file_path: Path for output CSV file
    """
    # Initialize NLTK resources
    initialize_nltk_resources()

    # Initialize processors
    text_processor = TextPreprocessor()
    dataset_processor = TaskDatasetProcessor(text_processor)
    reporter = AnalysisReporter()

    try:
        # Load and process dataset
        logger.info("Starting dataset processing...")
        raw_dataframe = dataset_processor.load_dataset(input_file_path)

        processed_dataframe = dataset_processor.process_descriptions(raw_dataframe)

        # Calculate statistics
        processing_stats = dataset_processor.calculate_statistics(processed_dataframe)

        # Save processed data
        dataset_processor.save_processed_data(processed_dataframe, output_file_path)

        # Generate analysis report
        reporter.print_analysis_report(processed_dataframe, processing_stats)

        logger.info("Processing completed successfully")

    except Exception as error:
        logger.error(f"Processing failed: {error}")
        raise


if __name__ == "__main__":
    main()