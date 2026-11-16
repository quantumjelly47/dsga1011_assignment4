import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "and", "or", "but", "you", "i", "he", "she", "it", "they",
        "we", "who", "what", "this", "that", "my", "your", "his", "her", "their", "our", "me", "him", "them", "us"
    }

    # personal keystroke mistakes
    keyboard = {
        "q": "wa", "w": "qase", "p": "o[", "n": "mb", "l": "k;", "m": "n,", "u": "yijh", "g": "h", "h": "g",
        "r": "te", "d": "sf", "e": "sdr"
    }

    def substitute(text):
        valid_positions = []

        for i in range(len(text)):
            if text[i].lower() in keyboard:
                valid_positions.append(i)

        if not valid_positions:
            return text

        idx = random.choice(valid_positions)
        char = text[idx].lower()

        replace = random.choice(keyboard[char])
        return text[:idx] + replace + text[idx+1:]
    
    words = example["text"].split()

    candidate_indices = [i for i, word in enumerate(words) if word.strip(".,!?;:\"'()").lower() 
                         not in stop_words and word.strip(".,!?;:\"'()").isalpha()]

    # chose a few words to replace
    words_replaced = 0
    num_replace = min(int(len(words) * 0.17), len(candidate_indices))
    # pick random word, making sure it has synonyms
    random.shuffle(candidate_indices)

    for word_idx in candidate_indices:
        original_word = words[word_idx]
        typo_type = random.choices(["substitute", "delete", "duplicate", "swap"], weights=[35, 30, 20, 15], k=1)[0]

        if len(original_word) < 5:
            continue

        if typo_type == "substitute":
            typo = substitute(original_word)
            if typo == original_word:
                continue

        elif typo_type == "delete":
            # make sure first/last never deleted
            pos = random.randint(1, len(original_word) - 2)
            typo = original_word[:pos] + original_word[pos+1:]

        elif typo_type == "duplicate":
            pos = random.randint(0, len(original_word) - 1)
            # don't want to duplicate capital letters punctuation
            if original_word[pos].isupper() or not original_word[pos].isalpha():
                continue
            typo = original_word[:pos] + original_word[pos] + original_word[pos:]

        else:
            pos = random.randint(0, len(original_word) - 2)
            # don't want to swap capital letters punctuation
            if original_word[pos].isupper() or not original_word[pos].isalpha():
                continue
            typo = original_word[:pos] + original_word[pos+1] + original_word[pos] + original_word[pos+2:]

        words[word_idx] = typo
        words_replaced += 1
        
        if words_replaced == num_replace:
            break

    # randomly join a few words together
    if len(words) > 3:
        num_merges = random.randint(0, 3)
        for i in range(num_merges):
            idx = random.randint(0, len(words) - 2)
            words[idx] = words[idx] + words[idx + 1]
            words.pop(idx + 1)

    example["text"] = " ".join(words)

    ##### YOUR CODE ENDS HERE ######

    return example