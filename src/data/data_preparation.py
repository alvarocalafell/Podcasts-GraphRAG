import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import logging
import re

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
    entities = [ent.text for ent in doc.ents if ent.label_ == 'PERSON' and len(ent.text.split()) > 1]
    return entities

def clean_entities(entities):
    # Function to clean individual entity
    def clean_entity(entity):
        # Remove leading/trailing whitespace and quotes
        entity = entity.strip().strip("'\"")
        # Remove possessive 's
        entity = re.sub(r"'s$", "", entity)
        # Remove filler words at the beginning
        entity = re.sub(r"^(the|a|an)\s+", "", entity, flags=re.IGNORECASE)
        return entity

    # Clean each entity
    cleaned_entities = [clean_entity(entity) for entity in entities if len(entity.split()) > 1]
    
    # Remove duplicates (case-insensitive) while preserving order
    seen = set()
    unique_entities = []
    for entity in cleaned_entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    return unique_entities

def topic_modeling(texts, n_topics=10):
    """Perform topic modeling using LDA."""
    logging.info(f"Performing topic modeling with {n_topics} topics...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_output = lda.fit_transform(doc_term_matrix)
    return lda, vectorizer, lda_output

def main():
    # Load the data
    data = load_data('data/toy_podcast_data.csv')
    logging.info(f"Loaded {len(data)} rows of data")

    # Preprocess the text
    logging.info("Preprocessing text...")
    tqdm.pandas(desc="Preprocessing")
    data['processed_text'] = data['text'].progress_apply(preprocess_text)

    # Perform NER
    logging.info("Performing Named Entity Recognition...")
    tqdm.pandas(desc="NER")
    data['entities'] = data['text'].progress_apply(perform_ner)

    # Clean entities
    logging.info("Cleaning entities...")
    tqdm.pandas(desc="Cleaning Entities")
    data['people'] = data['entities'].progress_apply(clean_entities)

    # Perform topic modeling
    lda, vectorizer, lda_output = topic_modeling(data['processed_text'])

    # Assign top topic to each document
    data['topic'] = lda_output.argmax(axis=1)

    # Create a list of top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = []
    for topic in lda.components_:
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        top_words_per_topic.append(top_words)

    # Assign top words of the main topic to each document
    data['topic_words'] = data['topic'].apply(lambda x: top_words_per_topic[x])

    # Select only the columns we want in the final processed file
    processed_data = data[['id', 'guest', 'title', 'text', 'processed_text', 'people']]
    # Save processed data
    logging.info("Saving processed data...")
    processed_data.to_csv('data/toy_processed_podcast_data.csv', index=False)
    logging.info("Processed data saved to 'toy_processed_podcast_data.csv'")

    # Print some results
    logging.info("Printing results...")
    print("\nFirst few rows of processed data:")
    print(processed_data.head())
    print("\nSample people from first row:")
    print(processed_data['people'].iloc[0][:5])
    print("\nTop words for each topic:")
    for topic_idx, topic_words in enumerate(top_words_per_topic):
        print(f"Topic {topic_idx}: {', '.join(topic_words)}")

    logging.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()