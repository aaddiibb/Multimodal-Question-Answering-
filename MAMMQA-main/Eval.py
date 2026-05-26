import os
import json
import pandas as pd
import re
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")

RE_ANSWER_TAG = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
RE_BEFORE_EXPLANATION = re.compile(r'(.*?)Explanation', re.IGNORECASE | re.DOTALL)


def normalize_answer(answer):
    return re.sub(r'\W+', ' ', answer.lower().strip()).split() if isinstance(answer, str) else []


def normalize_answer_spacy(answer):
    if isinstance(answer, str):
        doc = nlp(answer.lower().strip())
        return [token.lemma_ for token in doc if token.is_alpha]
    return []


def semantic_match(true_answer, predicted_answer, normalizer):
    true_count = Counter(normalizer(true_answer))
    pred_count = Counter(normalizer(predicted_answer))
    return all(pred_count[word] >= count for word, count in true_count.items())


def extract_relevant_answer(text):
    if not isinstance(text, str):
        return text
    if (m := RE_ANSWER_TAG.search(text)):
        return m.group(1).strip()
    if (m := RE_BEFORE_EXPLANATION.search(text)):
        return m.group(1).strip()
    if (idx := text.lower().find("answer to the question:")) != -1:
        return text[idx + len("answer to the question:"):].strip()
    return text.strip()


def get_tot(data):
    try:
        parsed_data = json.loads(data)
        return parsed_data['highest_rated_thought']['thought']
    except (json.JSONDecodeError, TypeError):
        return None


def evaluate_semantic_accuracy(df, our_results=False, use_spacy=False):
    df = df.rename(columns={
        'answer': 'true_answers',
        'answers': 'true_answers',
        'cot_response': 'predicted_answers',
        'Final Answers': 'predicted_answers',
        'Final Answer': 'predicted_answers',
        'response': 'predicted_answers',
        'modalities': 'modality'
    })
    df = df[['true_answers', 'predicted_answers', 'modality']].dropna()

    predicted_raw = df['predicted_answers'].astype(str).values
    true_raw = df['true_answers'].astype(str).values
    modalities = df['modality'].astype(str).values

    predicted_cleaned = predicted_raw if our_results else [extract_relevant_answer(ans) for ans in predicted_raw]

    strip_simplify = lambda x: x.strip("[]").replace("'", "").lower()
    predicted_simplified = list(map(strip_simplify, predicted_cleaned))
    true_simplified = list(map(strip_simplify, true_raw))

    normalizer = normalize_answer_spacy if use_spacy else normalize_answer

    semantic_results = [
        semantic_match(t, p, normalizer)
        for t, p in zip(true_simplified, predicted_simplified)
    ]

    df_result = pd.DataFrame({
        'modality': modalities,
        'semantically_correct': semantic_results
    })

    overall_accuracy = sum(semantic_results) / len(semantic_results)
    modality_perf = df_result.groupby('modality')['semantically_correct'].mean().to_dict()

    return {
        'Overall Semantic Accuracy': overall_accuracy,
        'Semantic Modality Performance': modality_perf
    }


def process_csv_file(file_path):
    try:
        json_path = os.path.splitext(file_path)[0] + ".json"
        if os.path.exists(json_path):
            return f"Skipped (already processed): {json_path}"

        our_results = "ours" in os.path.basename(file_path).lower()
        use_spacy = True

        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep="^")

        if "ToT" in file_path:
            df['predicted_answers'] = df.response.apply(get_tot)

        result = evaluate_semantic_accuracy(df, our_results=our_results, use_spacy=use_spacy)

        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        return f"Saved result: {json_path}"
    except Exception as e:
        return f"Failed to process {file_path}: {e}"



if __name__ == "__main__":
    base_dirs = ["/Users/krishnasinghrajput/Desktop/Agents/Gemini/qwen/", 
                 "/Users/krishnasinghrajput/Desktop/Agents/Gemini/qwen/"]

    all_csv_files = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".csv"):
                    all_csv_files.append(os.path.join(root, file))


    for file_path in all_csv_files:
        print(f"Processing {file_path}")
        result = process_csv_file(file_path)
        print(result)
