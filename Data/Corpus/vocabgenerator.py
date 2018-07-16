# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

from nltk.tokenize import word_tokenize

AUG0_FOLDER = "Augment0"
AUG1_FOLDER = "Augment1"
AUG2_FOLDER = "Augment2"

OUTFOLDER = "Outfiles"
DATA_FILE = os.path.join(OUTFOLDER, "data_file.txt")  # final data file
VOCAB_FILE = os.path.join("vocab.txt")  # vocabulary file saved in Corpus directory
FREQ_FILE = os.path.join(OUTFOLDER, "freq_dist.txt")  # pipe-delimted vocabulary frequency
SEQ_LEN_FILE = os.path.join(OUTFOLDER, "seq_len_file.txt")  # pipe-delimted sequence lengths > 50

DEFAULT_SEQ_LEN = 50  # Default max sequence length defined as 50


def generate_vocab_file(corpus_dir):
    """
    Generate the vocab.txt file for the training and prediction/inference. 
    Manually remove the empty bottom line in the generated file.
    """
    data_list = []
    vocab_list = []
    freq_dist = {}
    seq_len_dict = {}

    # Special tokens, with IDs: 0, 1, 2
    for t in ['_unk_', '_bos_', '_eos_']:
        vocab_list.append(t)

    # The word following this punctuation should be capitalized in the prediction output.
    for t in ['.', '!', '?']:
        vocab_list.append(t)

    # The word following this punctuation should not precede with a space in the prediction output.
    for t in ['(', '[', '{', '``', '$']:
        vocab_list.append(t)

    for fd in range(2, -1, -1):
        if fd == 0:
            file_dir = os.path.join(corpus_dir, AUG0_FOLDER)
        elif fd == 1:
            file_dir = os.path.join(corpus_dir, AUG1_FOLDER)
        else:
            file_dir = os.path.join(corpus_dir, AUG2_FOLDER)

        for data_file in sorted(os.listdir(file_dir)):
            full_path_name = os.path.join(file_dir, data_file)
            if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                with open(full_path_name, 'r') as f:
                    for line in f:
                        l = line.strip()
                        if not l:
                            # If skipped, we still need to write it to the final file
                            data_list.append(l)
                            continue
                        if l.startswith("Q:") or l.startswith("A:"):
                            # Tokenize (excluding Q/A)
                            tokens = word_tokenize(l[2:])

                            # Store tokenized string (including Q/A)
                            token_str = l[:2] + ' ' + ' '.join(tokens)
                            data_list.append(token_str)

                            # Cache long sequences
                            n = len(tokens)
                            if n > DEFAULT_SEQ_LEN:
                                seq_len_dict[token_str] = n

                            # Handle tokens for vocabulary
                            for t in tokens:
                                if len(t) and t != ' ':
                                    # Add token to vocabulary
                                    if t not in vocab_list:
                                        vocab_list.append(t)
                                    # If token is in vocabulary, increment its frequency
                                    if t in freq_dist.keys():
                                        freq_dist[t] += 1
                                    else:  # Otherwise add it and set to 1
                                        freq_dist[t] = 1

    print("Vocab size after all base data files scanned: {}".format(len(vocab_list)))

    # clear generated files from prior runs and create blanks
    for f in [DATA_FILE, VOCAB_FILE, FREQ_FILE, SEQ_LEN_FILE]:
        if os.path.isfile(f):
            os.remove(f)
        if not os.path.isfile(f):
            open(f, 'w').close()

    # Write objects to files (could be abstracted but more clear this way)
    with open(DATA_FILE, 'a') as f_out:
        for line in data_list:
            f_out.write(f"{line}\n")
    with open(VOCAB_FILE, 'a') as f_voc:
        for v in vocab_list:
            f_voc.write(f"{v}\n")
    with open(FREQ_FILE, 'a') as f_freq:
        for t, f in freq_dist.items():
            f_freq.write(f"{t}|{f}\n")
    with open(SEQ_LEN_FILE, 'w') as f_seq:
        for seq, n in seq_len_dict.items():
            f_seq.write(f"{seq}|{n}\n")

    print("The final vocab file generated. Vocab size: {}".format(len(vocab_list)))


if __name__ == "__main__":
    # from settings import PROJECT_ROOT  # can't import settings module

    # point to Corpus location
    corp_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(corp_dir)

    # make sure that the output folder exists for generated output 
    out_folder = os.path.join(corp_dir, OUTFOLDER)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    generate_vocab_file(corp_dir)
