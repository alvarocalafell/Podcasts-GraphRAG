import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
logging.info("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
logging.info("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')

def load_data(file_path):
    """Load the dataset from a CSV file."""
    logging.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_text(text):
    """Preprocess the text: tokenize, remove stopwords, lemmatize."""
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def perform_ner(text):
    """Perform Named Entity Recognition."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def topic_modeling(texts, n_topics=10):
    """Perform topic modeling using LDA."""
    logging.info(f"Performing topic modeling with {n_topics} topics...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    return lda, vectorizer

def main():
    # Load the data
    data = load_data('data/podcastdata_dataset.csv')
    logging.info(f"Loaded {len(data)} rows of data")
    
    # Preprocess the text
    logging.info("Preprocessing text...")
    tqdm.pandas(desc="Preprocessing")
    data['processed_text'] = data['text'].progress_apply(preprocess_text)
    
    # Perform NER
    logging.info("Performing Named Entity Recognition...")
    tqdm.pandas(desc="NER")
    data['entities'] = data['text'].progress_apply(perform_ner)
    
    # Perform topic modeling
    lda, vectorizer = topic_modeling(data['processed_text'])
    
    # Print some results
    logging.info("Printing results...")
    print("\nFirst few rows of processed data:")
    print(data.head())
    print("\nSample entities from first row:")
    print(data['entities'].iloc[0][:5])
    
    # Print top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        print(f"\nTop words for topic {topic_idx}:")
        print(", ".join(top_words))
    
    logging.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()