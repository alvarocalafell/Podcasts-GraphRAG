import pandas as pd

# Read the original CSV file
df = pd.read_csv('data/toy_processed_podcast_data.csv')

# Define the new data
new_data = {
    'Max Tegmark': {
        'organizations': ['MIT', 'Future of Humanity Institute', 'OpenAI'],
        'technologies': ['Deep Neural Networks', 'Quantum Computing', 'Brain-Computer Interfaces'],
        'topics': ['Artificial General Intelligence (AGI)', 'AI Safety and Ethics', 'Future of Technology', 'Consciousness and AI', 'Philosophy of Mind and AI']
    },
    'Christof Koch': {
        'organizations': ['Allen Institute for Brain Science', 'MIT', 'Stanford University'],
        'technologies': ['Brain-Computer Interfaces', 'Computer Vision Systems', 'Quantum Computing'],
        'topics': ['Consciousness and AI', 'Neuroscience and Cognition', 'Computational Neuroscience', 'Philosophy of Mind and AI', 'Human-AI Interaction']
    },
    'Steven Pinker': {
        'organizations': ['Harvard University', 'MIT', 'Google'],
        'technologies': ['Natural Language Processing Models', 'Deep Neural Networks', 'Big Data Analytics'],
        'topics': ['Artificial General Intelligence (AGI)', 'Human-AI Interaction', 'Natural Language Processing', 'AI Safety and Ethics', 'Neuroscience and Cognition']
    },
    'Yoshua Bengio': {
        'organizations': ['MILA', 'Google', 'Facebook AI Research'],
        'technologies': ['Deep Neural Networks', 'Convolutional Neural Networks', 'Recurrent Neural Networks', 'TensorFlow', 'PyTorch'],
        'topics': ['Deep Learning', 'Machine Learning Fundamentals', 'Natural Language Processing', 'Artificial General Intelligence (AGI)', 'AI in Business and Industry']
    },
    'Vladimir Vapnik': {
        'organizations': ['Facebook AI Research', 'Columbia University', 'AT&T'],
        'technologies': ['Support Vector Machines', 'Deep Neural Networks', 'Big Data Analytics'],
        'topics': ['Statistical Learning Theory', 'Machine Learning Fundamentals', 'Deep Learning', 'Computational Neuroscience', 'AI Safety and Ethics']
    },
    'Guido van Rossum': {
        'organizations': ['Python Software Foundation', 'Dropbox', 'Google'],
        'technologies': ['Python Programming Language', 'Cloud Computing Platforms', 'Big Data Analytics'],
        'topics': ['Programming Languages', 'Software Development Practices', 'Open Source Development', 'Technological Innovation', 'Online Communities and Knowledge Sharing']
    },
    'Jeff Atwood': {
        'organizations': ['Stack Overflow', 'Microsoft', 'Google'],
        'technologies': ['Cloud Computing Platforms', 'Big Data Analytics', 'Internet of Things'],
        'topics': ['Online Communities and Knowledge Sharing', 'Software Development Practices', 'Programming Languages', 'Open Source Development', 'Technological Innovation']
    },
    'Eric Schmidt': {
        'organizations': ['Google', 'MIT', 'Stanford University'],
        'technologies': ['Deep Neural Networks', 'Natural Language Processing Models', 'Cloud Computing Platforms', 'Internet of Things'],
        'topics': ['AI in Business and Industry', 'Technological Innovation', 'Future of Technology', 'Machine Learning Fundamentals', 'Human-AI Interaction']
    },
    'Stuart Russell': {
        'organizations': ['UC Berkeley', 'OpenAI', 'DeepMind'],
        'technologies': ['Reinforcement Learning Algorithms', 'Deep Neural Networks', 'Robotics Control Systems'],
        'topics': ['AI Safety and Ethics', 'Artificial General Intelligence (AGI)', 'Future of Technology', 'Philosophy of Mind and AI', 'Human-AI Interaction']
    },
    'Pieter Abbeel': {
        'organizations': ['UC Berkeley', 'OpenAI', 'Google'],
        'technologies': ['Reinforcement Learning Algorithms', 'Deep Neural Networks', 'Robotics Control Systems', 'PyTorch'],
        'topics': ['Reinforcement Learning', 'Deep Learning', 'Robotics and Automation', 'Machine Learning Fundamentals', 'AI in Business and Industry']
    }
}

# Function to get data for a guest
def get_data(guest):
    return new_data.get(guest, {'organizations': [], 'technologies': [], 'topics': []})

# Add new columns
df['organizations'] = df['guest'].apply(lambda x: get_data(x)['organizations'])
df['technologies'] = df['guest'].apply(lambda x: get_data(x)['technologies'])
df['topics'] = df['guest'].apply(lambda x: get_data(x)['topics'])

# Save the updated DataFrame to a new CSV file
df.to_csv('data/updated_toy_processed_dataset.csv', index=False)

print("Updated dataset saved as 'updated_toy_processed_dataset.csv'")