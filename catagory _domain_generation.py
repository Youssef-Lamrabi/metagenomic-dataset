import os
import json
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_API_BASE", "http://10.52.88.105:1234/v1"),
    
)
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss-20b")

INPUT_FILE = "combined_cleaned_dataset.jsonl" 
OUTPUT_FILE = "dataset_final_enriched.jsonl"
AUTOSAVE_INTERVAL = 50 


SUGGESTED_CATEGORIES = [
    "poultry_health_production", "immunology_host", "metabolism_functional", 
    "microbiome_analysis", "sequencing_bioinfo", "genomics_infra", 
    "alignment", "assembly", "binning", "annotation", "taxonomy", 
    "diversity_analysis", "quantification", "pipeline_design"
]

SUGGESTED_TOPICS = [
    "QC & Preprocessing", "Assembly", "Binning", "Annotation", "Taxonomy / Classification", 
    "Metagenomics Tools", "Errors & Debugging", "Performance & Scaling", "Gut Microbiota", 
    "Metabolites / SCFA", "Fermentation", "Diet / Nutrition", "Poultry Biology", 
    "Immunology", "Receptor Signaling", "Gut Barrier / Mucosal", "Energy / Metabolism", "Epigenetics"
]

def get_system_prompt():
    return f"""You are an expert bioinformatician and biologist data annotator.
Your task is to classify a given scientific text into EXACTLY ONE Category and EXACTLY ONE primary Topic.

Here are some SUGGESTED Categories: {SUGGESTED_CATEGORIES}
Here are some SUGGESTED Topics: {SUGGESTED_TOPICS}

RULES:
1. You may choose from the suggested lists, OR you can INVENT a new short, descriptive Category or Topic if the text does not fit any of the suggestions well.
2. Keep your invented Categories and Topics concise (1 to 3 words max, e.g., "viral_genomics" or "Host-Pathogen Interaction").
3. Output your response STRICTLY as a raw JSON object. No markdown, no explanations.

EXPECTED JSON FORMAT:
{{
    "category": "your_chosen_or_invented_category",
    "topic": "your_chosen_or_invented_topic"
}}"""


def classify_with_llm(text):
    try:
       
        truncated_text = text[:5000] if isinstance(text, str) else str(text)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Text to classify:\n{truncated_text}"}
            ],
            temperature=0.3, 
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            parsed_data = json.loads(match.group(0))
            
            
            cat = parsed_data.get('category', 'other')
            topic = parsed_data.get('topic', 'Other / Unclassified')
            
            
            if isinstance(cat, str):
                cat = cat.lower().replace(" ", "_")
            if isinstance(topic, str):
                topic = topic.title()
                
            return cat, topic
            
    except Exception as e:
       
        pass
        
    return None, None

def main():
    
    if not os.path.exists(INPUT_FILE):
        print(f"ichier {INPUT_FILE} introuvable.")
        return

    
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    df = pd.DataFrame(records)

 
    if 'topic_main' not in df.columns:
        df['topic_main'] = 'Other / Unclassified'
        

    mask_cat = df['category'].isin(['other', 'Other', None, '']) | df['category'].isna()
    mask_topic = df['topic_main'].isin(['Other / Unclassified', None, '']) | df['topic_main'].isna()
    
    target_indices = df[mask_cat | mask_topic].index.tolist()
    print(f"{len(target_indices)} has value missing .")

  
    processed_count = 0
    try:
        for idx in tqdm(target_indices):
            text = str(df.loc[idx, 'full_text'])
            new_cat, new_topic = classify_with_llm(text)
            
            if new_cat is not None and new_topic is not None:
               
                if df.loc[idx, 'category'] in ['other', 'Other', None, ''] or pd.isna(df.loc[idx, 'category']):
                    df.loc[idx, 'category'] = new_cat
                    
                if df.loc[idx, 'topic_main'] in ['Other / Unclassified', None, ''] or pd.isna(df.loc[idx, 'topic_main']):
                    df.loc[idx, 'topic_main'] = new_topic
                
            processed_count += 1
            
            
            if processed_count % AUTOSAVE_INTERVAL == 0:
                df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)
                
    except KeyboardInterrupt:
        print("\n Processus interrompu par l'utilisateur.")
    finally:
        print(f"\n Sauvegarde finale du dataset ")
        df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)
        print(f" fin : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()