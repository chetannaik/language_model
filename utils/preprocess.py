import os
import random
from nltk import sent_tokenize, word_tokenize

data_path = os.path.join('/Users/chetan/s/1/language_model', 'data')
input_file = os.path.join(data_path, 'input.txt')
processed_file = os.path.join(data_path, 'p_input.txt')
train_file_path = os.path.join(data_path, 'train_data.txt')
vocabulary_file_path = os.path.join(data_path, 'vocabulary.txt')

WINDOW_SIZE = 5
MID = WINDOW_SIZE/2


def generate_training_data(vocabulary):
    with open(processed_file) as fp:
        with open(train_file_path, 'w') as tfp:
            for sentence in fp:
                words = sentence.split()
                words = ['$START$'] + words + ['$END$']

                for idx in range(0, len(words) - WINDOW_SIZE + 1):
                    pos_window_words = words[idx: idx + WINDOW_SIZE]

                    main_word = pos_window_words[MID]
                    random_word = main_word

                    while random_word == main_word:
                        random_word = random.choice(tuple(vocabulary))

                    neg_window_words = pos_window_words[:MID] + [random_word] + pos_window_words[MID+1:]

                    for word in pos_window_words:
                        tfp.write(word+" ")
                    tfp.write('\n')

                    for word in neg_window_words:
                        tfp.write(word+" ")
                    tfp.write('\n')


def get_vocabulary():
    vocabulary = set()
    with open(input_file) as f:
        filelines = f.read()
        filelines = filelines.replace('\r\n', ' ')
        filelines = filelines.replace('\n', ' ')
        sentences = sent_tokenize(filelines.decode('utf-8').strip())
        with open(processed_file, 'w') as fp:
            for sentence in sentences:
                words = word_tokenize(sentence)
                for word in words:
                    word = word.encode('utf-8')
                    fp.write(word + " ")
                    vocabulary.add(word)
                fp.write('\n')
    return vocabulary


def save_vocabulary(vocabulary):
    with open(vocabulary_file_path, 'w') as fp:
        for word in vocabulary:
            fp.write(word+"\n")

if __name__ == '__main__':
    vocabulary = get_vocabulary()
    generate_training_data(vocabulary)
    vocabulary.add('$START$')
    vocabulary.add('$END$')
    vocabulary.add('UNK')
    save_vocabulary(vocabulary)
