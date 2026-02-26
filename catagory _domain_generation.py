import os
import json
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
base_url="http://10.52.88.105:1234/v1",
api_key="sk-xxx"
)
MODEL_NAME =  "gpt-oss-20b"


INPUT_FILE = "combined_cleaned_dataset.jsonl" 
OUTPUT_FILE = "dataset_final_enriched.jsonl"
AUTOSAVE_INTERVAL = 50

SUGGESTED_CATEGORIES = [
    # Catégories réellement présentes dans le dataset
    "alignment", "annotation", "assembly", "assembly_qc", "bin_qc",
    "binning", "diversity_analysis", "functional_profiling", "genomics_infra",
    "host_decontamination", "machine_learning", "multiomics", "pipeline_design",
    "qc_preprocessing", "quantification", "sequencing", "statistical_analysis",
    "taxonomy", "visualization"
]

SUGGESTED_DOMAINS = [
    # Domaines réellement présents dans le dataset
    "QC & Preprocessing", "Assembly", "Binning", "Annotation",
    "Taxonomy / Classification", "Metagenomics Tools", "Errors & Debugging",
    "Performance & Scaling", "Gut Microbiota", "Metabolites / SCFA",
    "Fermentation", "Diet / Nutrition", "Poultry Biology",
    # Domaines manquants ajoutés
    "Energy / Metabolism", "Epigenetics", "Gut Barrier / Mucosal",
    "Immunology", "Receptor Signaling"
]

def get_system_prompt():
    return f"""You are an expert bioinformatician and biologist data annotator.
Your task is to classify a given scientific text into EXACTLY ONE Category and EXACTLY ONE primary Domain.

Here are some SUGGESTED Categories: {SUGGESTED_CATEGORIES}
Here are some SUGGESTED Domains: {SUGGESTED_DOMAINS}

RULES:
1. You may choose from the suggested lists, OR INVENT a new short, descriptive Category/Domain if needed.
2. Output your response STRICTLY as a raw JSON object.

EXPECTED JSON FORMAT:
{{
    "category": "your_category",
    "domain": "Your Domain"
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
            domain = parsed_data.get('domain', 'Other / Unclassified')
            
            if isinstance(cat, str): cat = cat.lower().replace(" ", "_")
            if isinstance(domain, str): domain = domain.title()
                
            return cat, domain
        else:
            print(f"  [WARN] Pas de JSON trouvé dans la réponse: {raw_output[:200]}")
            
    except Exception as e:
        print(f"  [ERROR] LLM: {type(e).__name__}: {e}")
    return None, None


def extract_text_from_row(row):
    parts = []
    for key in ['instruction', 'output', 'question', 'answer', 'steps', 'constraints']:
        if pd.notna(row.get(key)):
            val = row[key]
            if isinstance(val, list):
                parts.append(" ".join(map(str, val)))
            else:
                parts.append(str(val))
    return " ".join(parts)


def is_invalid_category(cat):
    return cat in ('other', 'Other', None, '') or not cat


def is_invalid_domain(domains):
    if not isinstance(domains, list):
        return True
    if len(domains) == 0:
        return True
    if 'Other / Unclassified' in domains:
        return True
    return False


def main():

    if not os.path.exists(INPUT_FILE):
        print("File introuvable.")
        return

    # 1. Lire toutes les lignes en JSON brut (préserve le format hétérogène)
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 2. Identifier les lignes à traiter
    target_indices = []
    for i, row in enumerate(rows):
        cat = row.get('category', '')
        domains = row.get('domains', [])
        if is_invalid_category(cat) or is_invalid_domain(domains):
            target_indices.append(i)

    print(f"{len(target_indices)} / {len(rows)} lignes à traiter.")

    # 3. Traiter et sauvegarder en préservant le format original
    processed_count = 0
    try:
        for idx in tqdm(target_indices):
            row = rows[idx]

            # Extraire le texte
            parts = []
            for key in ['instruction', 'output', 'question', 'answer', 'steps', 'constraints']:
                val = row.get(key)
                if val is not None and str(val).strip():
                    if isinstance(val, list):
                        parts.append(" ".join(map(str, val)))
                    else:
                        parts.append(str(val))
            text = " ".join(parts).strip()

            if not text:
                continue

            new_cat, new_domain = classify_with_llm(text)

            if new_cat is not None and new_domain is not None:
                updated = False
                # Bloc 1 : Corriger la catégorie si invalide
                if is_invalid_category(row.get('category', '')):
                    row['category'] = new_cat
                    updated = True

                # Bloc 2 : Corriger le domaine INDÉPENDAMMENT
                if is_invalid_domain(row.get('domains', [])):
                    row['domains'] = [new_domain]
                    updated = True

                if updated:
                    print(f"  [OK] idx={idx} → cat={new_cat}, domain={new_domain}")
                else:
                    print(f"  [SKIP] idx={idx} — déjà valide")
            else:
                print(f"  [FAIL] idx={idx} — LLM n'a rien retourné")

            processed_count += 1

            # Autosave toutes les N lignes (format original préservé)
            if processed_count % AUTOSAVE_INTERVAL == 0:
                save_jsonl(rows, OUTPUT_FILE)

    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        save_jsonl(rows, OUTPUT_FILE)
        print(f"Fichier généré : {OUTPUT_FILE}")


def save_jsonl(rows, filepath):
    """Sauvegarde ligne par ligne — chaque objet garde SES propres champs uniquement."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()