import numpy as np
import re


def split_into_words(text):
    words = []
    for word in re.split(' |\t', text.strip()):
        if len(word) > 0:
            words.append(word)

    return words


def compute_amount_of_words(text):
    return len(split_into_words(text))


def tokens_per_text_mean(tokenized_texts: list) -> float:
    return np.mean([len(tokenized_text) for tokenized_text in tokenized_texts])


def tokens_per_text_quantile(tokenized_texts: list, quantile: float) -> float:
    return np.quantile([len(tokenized_text) for tokenized_text in tokenized_texts], q=quantile)


def tokens_per_symbol(texts: list, tokenized_texts: list) -> float:
    if len(texts) != len(tokenized_texts):
        raise Exception(f"Amount of texts: {len(texts)} != Amount of tokenized texts {len(tokenized_texts)}")

    return np.mean([len(tokenized_text) / len(text) for text, tokenized_text in zip(texts, tokenized_texts)])


def tokens_per_word(texts: list, tokenized_texts: list) -> float:
    if len(texts) != len(tokenized_texts):
        raise Exception(f"Amount of texts: {len(texts)} != Amount of tokenized texts {len(tokenized_texts)}")

    return np.mean([len(tokenized_text) / compute_amount_of_words(text) for text, tokenized_text in zip(texts, tokenized_texts)])


def out_of_vocabulary_words_percent(vocabulary: list, texts: list) -> float:
    vocabulary_set = set(vocabulary)

    word_amount = 0
    not_in_vocabulary_amount = 0

    for text in texts:
        for word in split_into_words(text):
            word_amount += 1
            if word not in vocabulary_set:
                not_in_vocabulary_amount += 1

    return not_in_vocabulary_amount / word_amount
