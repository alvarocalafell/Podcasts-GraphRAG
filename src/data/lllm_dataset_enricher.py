import xml.etree.ElementTree as ET
import openai
import json

# Set up your OpenAI API key
openai.api_key = 'your-api-key-here'

def process_interview_with_llm(content, title):
    prompt = f"""
    Analyze the following interview transcript and extract the following information:

    1. Entities:
       - People (researchers, scientists, historical figures)
       - Organizations (universities, companies, research labs)
       - Technologies (specific AI techniques, algorithms, hardware)
       - Concepts (theoretical ideas, research areas)

    2. Relationships:
       - Person works at Organization
       - Person developed Technology
       - Concept is related to Technology
       - Person collaborates with Person

    3. Main Topics: Identify the top 5 main themes or topics discussed in the interview.

    Interview Title: {title}

    Transcript:
    {content[:4000]}  # Truncating to 4000 chars due to token limits. Adjust as needed.

    Provide the results in the following JSON format:
    {{
        "entities": {{
            "people": [],
            "organizations": [],
            "technologies": [],
            "concepts": []
        }},
        "relationships": [],
        "main_topics": []
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant skilled in analyzing interview transcripts and extracting relevant information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return json.loads(response.choices[0].message['content'])

# Parse the XML file
tree = ET.parse('toy_dataset.xml')
root = tree.getroot()

# Process each document
for document in root.findall('.//document'):
    title = document.find('title').text
    content = document.find('document_content').text
    
    print(f"Processing: {title}")
    
    results = process_interview_with_llm(content, title)
    
    print("Entities:")
    for category, items in results['entities'].items():
        print(f"  {category.capitalize()}: {', '.join(items)}")
    
    print("\nRelationships:")
    for relationship in results['relationships']:
        print(f"  {relationship}")
    
    print("\nMain Topics:")
    for topic in results['main_topics']:
        print(f"  {topic}")
    
    print("\n" + "="*50 + "\n")