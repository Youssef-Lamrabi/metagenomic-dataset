import requests
import json
import re
import time
import os
import logging
from datetime import datetime
import dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


dotenv.load_dotenv()


GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')

HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

OUTPUT_FILE = 'dataset_metagenomics_troubleshooting.jsonl'
MAX_ISSUES_PER_REPO = 1000 


GITHUB_REPOS = {
    
    #'MetaSPAdes': 'ablab/spades',
    #'MEGAHIT': 'voutcn/megahit',
    'CheckM': 'Ecogenomics/CheckM',
    'CheckM2': 'chklovski/CheckM2',
    
    'FastP': 'OpenGene/fastp',
    'FastQC': 's-andrews/FastQC',
    'Trimmomatic': 'usadellab/Trimmomatic',
    'MultiQC': 'MultiQC/MultiQC',
    
    'Kraken2': 'DerrickWood/kraken2',
    'Bowtie2': 'BenLangmead/bowtie2',
    'BWA': 'lh3/bwa',
    'Samtools': 'samtools/samtools',
    'QUAST': 'ablab/quast',
    'Micromamba': 'mamba-org/micromamba'
}


def clean_text(text):
    if not text: return ''
    
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
   
    text = re.sub(r' {2,}', ' ', text).strip()
    return text[:3000] 

def score_comment(comment):
    """
    Évalue la probabilité qu'un commentaire soit LA bonne solution.
    """
    score = 0
    body = comment.get('body', '').lower()
    
 
    author_role = comment.get('author_association', '')
    if author_role in ['OWNER', 'MEMBER', 'COLLABORATOR']:
        score += 50
        
    
    if '```' in body or '`' in body:
        score += 20
        
  
    if any(kw in body for kw in ['solution', 'fixed in', 'try this', 'resolved', 'workaround', 'use this flag']):
        score += 15
        
   
    reactions = comment.get('reactions', {})
    score += (reactions.get('+1', 0) * 10)
    score += (reactions.get('heart', 0) * 10)
    score += (reactions.get('hooray', 0) * 10)
    
    return score

def handle_rate_limit(response):
    """Met le script en pause si on atteint la limite de l'API GitHub."""
    remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
    if remaining < 10:
        reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
        sleep_time = max(reset_time - int(time.time()) + 5, 0)
        logging.warning(f" Limite d'API atteinte. Pause de {sleep_time} secondes...")
        time.sleep(sleep_time)


def extract_issues(tool_name, repo):
    records = []
    page = 1
    logging.info(f"extraction pour [{tool_name}] ({repo})...")

    while len(records) < MAX_ISSUES_PER_REPO:
        url = f'https://api.github.com/repos/{repo}/issues'
        
        params = {'state': 'closed', 'per_page': 50, 'page': page, 'sort': 'updated'}
        
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
            handle_rate_limit(resp)
            
            if resp.status_code != 200:
                logging.error(f"Erreur API {resp.status_code} pour {repo}")
                break

            issues = resp.json()
            if not issues: 
                break 

            for issue in issues:
                
                if 'pull_request' in issue: 
                    continue

                title = clean_text(issue.get('title', ''))
                body = clean_text(issue.get('body', ''))
                issue_url = issue.get('html_url', '')

                
                if not title or not body or len(body.split()) < 15:
                    continue

                
                if issue.get('comments', 0) == 0:
                    continue
                    
                comments_url = issue.get('comments_url', '')
                c_resp = requests.get(comments_url, headers=HEADERS, timeout=10)
                handle_rate_limit(c_resp)
                
                if c_resp.status_code != 200:
                    continue
                    
                comments = c_resp.json()
                if not comments:
                    continue

                
                best_comment = max(comments, key=score_comment)
                output_text = clean_text(best_comment.get('body', ''))

                
                if len(output_text.split()) < 10:
                    continue

                
                record = {
                    "instruction": f"Solve this bioinformatics issue regarding {tool_name}: {title}",
                    "input": body,
                    "output": output_text,
                    "metadata": {
                        "tool": tool_name,
                        "source": "github_issues",
                        "url": issue_url
                    }
                }
                
                records.append(record)
                
                
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        except Exception as e:
            logging.error(f"Erreur lors du traitement de {repo}: {str(e)}")
            time.sleep(5)

        page += 1
        time.sleep(0.5) 

    logging.info(f" [{tool_name}] terminé : {len(records)} exemples de troubleshooting extraits.")
    return records


if __name__ == "__main__":
    print(f" Lancement de la collecte de Troubleshooting GitHub...")
    
   
    if not os.path.exists(OUTPUT_FILE):
        open(OUTPUT_FILE, 'w').close()
        
    total_extracted = 0
    
    for tool, repo in GITHUB_REPOS.items():
        recs = extract_issues(tool, repo)
        total_extracted += len(recs)
        
    print(f"\n  {total_extracted} exemples d'erreurs réelles ajoutés au dataset.")
    print(f" Fichier généré : {OUTPUT_FILE}")