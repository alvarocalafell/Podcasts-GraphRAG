# Lex Fridman Podcast Analysis using GraphRAG

This project aims to analyze and enhance Lex Fridman podcast transcripts using Graph Retrieval-Augmented Generation (GraphRAG) technology. It leverages natural language processing, graph-based knowledge representation, and retrieval-augmented generation to provide context-rich analysis and content enrichment.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Data Preparation](#data-preparation)
6. [Knowledge Graph Construction](#knowledge-graph-construction)
7. [GraphRAG Implementation](#graphrag-implementation)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

The Lex Fridman Podcast Analysis project processes podcast transcripts to:

- Extract key entities and topics
- Build a knowledge graph representing relationships between concepts discussed in the podcast
- Implement a GraphRAG system for enhanced transcript analysis and content enrichment

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/lex-fridman-graphrag.git
   cd lex-fridman-graphrag
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the necessary NLTK data and spaCy model:
   ```
   python -m nltk.downloader punkt stopwords wordnet
   python -m spacy download en_core_web_sm
   ```

## Usage

To prepare the data and perform initial analysis:

```
python data_preparation.py
```

(Add more usage instructions as you develop other scripts)

## Project Structure

```
lex-fridman-graphrag/
│
├── data/
│   └── podcastdata_dataset.csv
│
├── src/
│   ├── data_preparation.py
│   ├── knowledge_graph.py
│   └── graphrag.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
└── README.md
```

## Data Preparation

The `data_preparation.py` script performs the following tasks:

1. Loads the podcast transcript data
2. Preprocesses the text (tokenization, stopword removal, lemmatization)
3. Performs Named Entity Recognition (NER)
4. Conducts topic modeling using Latent Dirichlet Allocation (LDA)

## Knowledge Graph Construction

(To be implemented)

This phase will involve:
- Extracting relationships between entities
- Building a graph structure representing the podcast content
- Storing the graph in a suitable database (e.g., Neo4j)

## GraphRAG Implementation

(To be implemented)

This phase will involve:
- Implementing a retrieval mechanism to query the knowledge graph
- Developing a generation model to produce enhanced analysis
- Creating a pipeline that combines retrieval and generation

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

