import json

def build_from_blimp_row(row: dict) -> str:
    sentences = [row['sentence_good'], row['sentence_bad']]
    results = []
    for idx, sentence in enumerate(sentences):
        prompt = sentence
        results.append({
            "dataset": "blimp",
            "prompt": prompt,
            "id": row['UID'] + '#' + row['pairID'] + '#' + str(idx),
            "linguistics_term": row['linguistics_term'],
            "label": 'bad' if idx == 1 else 'good'
        })
    return results