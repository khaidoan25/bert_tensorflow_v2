"""Tensorflow BERT model."""

import tensorflow as tf
import math
import json
import copy
import six
from tensorflow.python.keras.layers import Layer, Embedding, Dropout, Dense

def gelu(x):
    """
        Implementation of the gelu activation.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly diferent results):
        0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    """

    return x * 0.5 * (1.0 + tf.math.erf(x / math.sqrt(2.0)))


class BertConfig():
    """
        Configuration class to store the configuration of 'BertModel'.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """
            Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of 'inputs_ids' in 'BertModel'.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of 'intermediate' (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: the non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the 'token_type_ids' passed into
                'BertModel'.
            initializer_range: The sttdev of the truncated_normal_initializer for
                iniitalizing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hiddne_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(Layer):
    def __init__(self, config, variance_epsilon=1e-12):
        """
            Construct a layernorm module in the TF style (epsilon inside the square root).
        """

        super().__init__()
        self.gamma = tf.Variable(tf.ones(config.hidden_size))
        self.beta = tf.Variable(tf.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_mean(tf.pow(x - mean, 2), axis=-1, keepdims=True)
        x = (x - mean) / tf.math.sqrt(std + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(Layer):
    def __init__(self, config):
        """
            Construct the embedding module from word, position and segment embeddings.
        """
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick
        # with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, segment_ids=None):

        # shape of input_ids (batch_size x seq_length)
        seq_length = input_ids.size(1)
        position_ids = tf.range(seq_length)
        position_ids = tf.broadcast_to(position_ids, input_ids.shape)
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        # sum over 3 types of embedding, follow by layernorm and dropout layer
        embeddings = word_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BERTSelfAttention(Layer):
    def __init__(self, config):
        super().__init__()
        


if __name__ == "__main__":
    pass