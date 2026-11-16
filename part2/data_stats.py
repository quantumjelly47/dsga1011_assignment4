import numpy as np
from load_data import T5Dataset

def get_raw_stats(nl_file, sql_file):
    sentences = []
    queries = []

    with open(nl_file, "r") as fnl, open(sql_file, "r") as fsql:
        for s_line, q_line in zip(fnl, fsql):
            # Raw whitespace word counts before any pre-processing or T5 tokenization
            sentences.append(s_line.strip().split())
            queries.append(q_line.strip().split())

    return {
        "number of examples": len(sentences),
        "mean sentence length": np.mean([len(s) for s in sentences]),
        "mean sql query length": np.mean([len(q) for q in queries]),
        "vocab size (nl)": len(set(w for s in sentences for w in s)),
        "vocab size (sql)": len(set(w for q in queries for w in q)),
    }

def get_preprocessed_stats(split):
    # Uses your exact load_data.py preprocessing + T5TokenizerFast (with add_special_tokens=True)
    ds = T5Dataset("data", split)

    enc_lens = [len(ids) for ids in ds.input_ids]
    dec_lens = [len(ids) for ids in ds.target_ids]
    enc_vocab = set(int(tok) for ids in ds.input_ids for tok in ids)
    dec_vocab = set(int(tok) for ids in ds.target_ids for tok in ids)

    return {
        "number of examples": len(ds),
        "mean sentence length": np.mean(enc_lens),
        "mean sql query length": np.mean(dec_lens),
        "vocab size (nl)": len(enc_vocab),
        "vocab size (sql)": len(dec_vocab),
    }

print("\nTABLE 1: BEFORE PREPROCESSING (RAW WORDS)")
train_raw = get_raw_stats("data/train.nl", "data/train.sql")
dev_raw = get_raw_stats("data/dev.nl", "data/dev.sql")
print(f"{'statistic name':<45} {'train':>12} {'dev':>12}")
for key in train_raw:
    print(f"{key:<45} {train_raw[key]:>12.2f} {dev_raw[key]:>12.2f}")

print("\nTABLE 2: AFTER PREPROCESSING (google-t5/t5-small)")
train_prep = get_preprocessed_stats("train")
dev_prep = get_preprocessed_stats("dev")
print(f"{'statistic name':<45} {'train':>12} {'dev':>12}")
for key in train_prep:
    print(f"{key:<45} {train_prep[key]:>12.2f} {dev_prep[key]:>12.2f}")