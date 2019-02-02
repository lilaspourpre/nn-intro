from yargy.tokenizer import MorphTokenizer
import nltk


def prepare_for_dataset(sentences):
    tokenizer = MorphTokenizer()
    sentences = [tokenize_sentence(tokenizer, sentence) for sentence in sentences]
    max_len = max([len(i) for i in sentences])
    words = list(set([word.lower() for sent in sentences for word in sent]))
    dataset, id2word, word2id = create_dataset(words, sentences)
    return dataset, id2word, word2id, max_len


def tokenize_sentence(tokenizer, sentence):
    tokens = tokenizer.split(sentence.replace("\n", ''))
    return tokens


def create_dataset(words, sentences):
    word2ind = {word: index for index, word in enumerate(words, start=1)}
    ind2word = {str(index): word for index, word in enumerate(words, start=1)}
    sentences_int = [[word2ind[w.lower()] for w in s] for s in sentences]
    word2ind["<PAD>"] = 0
    ind2word[0] = "<PAD>"
    return list(nltk.bigrams(sentences_int)), ind2word, word2ind
