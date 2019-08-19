import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__),"../")))
print(sys.path)
from tqdm import tqdm
import tensorflow as tf
import tokenization as tokenization
import numpy as np
from modelling import *
from processor import (Semeval_NLI_B_Processor, Semeval_NLI_M_Processor,
                    Semeval_QA_B_Processor, Semeval_QA_M_Processor,
                    Semeval_single_Processor, Sentihood_NLI_B_Processor,
                    Sentihood_NLI_M_Processor, Sentihood_QA_B_Processor,
                    Sentihood_QA_M_Processor, Sentihood_single_Processor)

processors = {
            "sentihood_single":Sentihood_single_Processor,
            "sentihood_NLI_M":Sentihood_NLI_M_Processor,
            "sentihood_QA_M":Sentihood_QA_M_Processor,
            "sentihood_NLI_B":Sentihood_NLI_B_Processor,
            "sentihood_QA_B":Sentihood_QA_B_Processor,
            "semeval_single":Semeval_single_Processor,
            "semeval_NLI_M":Semeval_NLI_M_Processor,
            "semeval_QA_M":Semeval_QA_M_Processor,
            "semeval_NLI_B":Semeval_NLI_B_Processor,
            "semeval_QA_B":Semeval_QA_B_Processor,
            }

class InputFeatures():
    """
        A single set of features of data.

        input_ids: index of token in vocabulary
        input_mask: separate sentence with padding
        segment_ids: separate 2 sentences
        label_id: label index
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
        Load a data file into a list of InputBatchs.
    """

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for index, example in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            # Check whether it is bert pair case or bert single case
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # BERT pair case
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # BERT single case
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            # BERT pair case
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))

    return features

def prepare_dataloaders(task_name, 
                        vocab_file,
                        data_dir,
                        do_lower_case=True,
                        train_batch_size=64,
                        num_train_epochs=100.0,
                        max_seq_length=100,
                        test=False,
                        eval_batch_size=8):
    
    processor = processors[task_name]()
    # Ex: ['None', 'Positive', 'Negative']
    label_list = processor.get_labels()

    # Define tokenizer (use for Vietnamese need to chang this)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case
    )

    # Prepare training set
    train_examples = processor.get_train_examples(data_dir)
    num_train_steps = int(
        len(train_examples) / train_batch_size * num_train_epochs
    )

    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer
    )

    all_input_ids = np.array([f.input_ids for f in train_features], dtype=np.int32)
    all_input_mask = np.array([f.input_mask for f in train_features], dtype=np.int32)
    all_segment_ids = np.array([f.segment_ids for f in train_features], dtype=np.int32)
    all_label_ids = np.array([f.label_id for f in train_features], dtype=np.int32)

    train_data = tf.data.Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids, all_label_ids)).shuffle(len(all_input_ids))
    train_data = train_data.batch(train_batch_size)

    if test:
        test_examples = processor.get_test_examples(data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, max_seq_length, tokenizer
        )

        all_input_ids = np.array([f.input_ids for f in test_features], dtype=np.int32)
        all_input_mask = np.array([f.input_mask for f in test_features], dtype=np.int32)
        all_segment_ids = np.array([f.segment_ids for f in test_features], dtype=np.int32)
        all_label_ids = np.array([f.label_id for f in test_features], dtype=np.int32)

        test_data = tf.data.Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids, all_label_ids))
        test_data = test_data.batch(eval_batch_size)
        return train_data, test_data
    return train_data

def model_for_classification(bert_config,
                             init_checkpoint,
                             len_label_list):

    model = BERTModel(bert_config)
    restore_checkpoint = tf.train.Checkpoint(model=model)
    status = restore_checkpoint.restore(init_checkpoint)

    f_input_ids = np.zeros((3, 100))
    f_segment_ids = np.zeros((3, 100))
    f_attention_mask = np.zeros((3, 100))
    f_labels = np.zeros((3,))
    f_labels = tf.cast(f_labels, tf.int32)
    model(f_input_ids, f_segment_ids, f_attention_mask)

    status.assert_consumed()

    # return model
    return model

    
if __name__ == "__main__":

    # Check whether gpu is available
    # device = tf.device("/device:GPU:0" if tf.test.is_gpu_available() else "/cpu:0")
    bert_config = BertConfig.from_json_file("D:/ABSA/ABSA-BERT-pair-test/uncased_L-12_H-768_A-12/bert_config.json")
    model = model_for_classification(bert_config, "D:/ABSA/ABSA-BERT-pair-test/uncased_L-12_H-768_A-12/bert_model.ckpt", 4)
    w = model.bert.get_weights()
    # train_data, test_data = prepare_dataloaders("semeval_NLI_M",
    #                                              "/home/ddkhai/Documents/ABSA/uncased_L-12_H-768_A-12/vocab.txt",
    #                                              "/home/ddkhai/Documents/ABSA/bert_tensorflow/data/semeval2014/bert-pair",
    #                                              test=True)

    # model.compile(tf.keras.optimizers.Adam(), )
    # model.evaluate(test_data)