"""Tensorflow BERT model."""

import tensorflow as tf
import math
import json
import copy
import six
from tensorflow.python.keras.layers import Embedding, Dropout, Dense, LayerNormalization
from tensorflow.python.keras import Model

def gelu(x):
    """
        Implementation of the gelu activation.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly diferent results):
        0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    """

    return x * 0.5 * (1.0 + tf.math.erf(x / math.sqrt(2.0)))

def create_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


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
            num_attention_heads: Number of attention heads for each attention layer in
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
        self.num_hidden_layers = num_hidden_layers
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


# class BERTLayerNorm(Model):
#     def __init__(self, config, variance_epsilon=1e-12):
#         """
#             Construct a layernorm module in the TF style (epsilon inside the square root).
#         """

#         super().__init__()
#         self.gamma = tf.Variable(tf.ones(config.hidden_size))
#         self.beta = tf.Variable(tf.zeros(config.hidden_size))
#         self.variance_epsilon = variance_epsilon

#     def call(self, x):
#         """
#             Calculate mean, std follow hidden_size dimension. Not like BatchNorm, which is
#             calculated follow batch size dimension
#             => Each sample has a particular (mean, std).

#             Args:
#                 x: tensor with shape batch_size x seq_length x hidden_size.
#         """

#         mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
#         std = tf.math.reduce_mean(tf.pow(x - mean, 2), axis=-1, keepdims=True)
#         x = (x - mean) / tf.math.sqrt(std + self.variance_epsilon)
#         return self.gamma * x + self.beta

class BERTEmbeddings(Model):
    def __init__(self, config):
        """
            Construct the embedding module from word, position and segment embeddings.
        """
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, embeddings_initializer=tf.initializers.TruncatedNormal(stddev=0.02))
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=tf.initializers.TruncatedNormal(stddev=0.02))
        self.segment_embeddings = Embedding(config.type_vocab_size, config.hidden_size, embeddings_initializer=tf.initializers.TruncatedNormal(stddev=0.02))

        # self.LayerNorm is not snake-cased to stick
        # with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNormalization()
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, segment_ids=None):
        """
            Args:
                input_ids: each entry is idex of that word in vocabulary.
                    shape: batch_size x seq_length x 
                segment_ids: segment sentence A vs sentence B.
        """

        seq_length = input_ids.shape[1]
        position_ids = tf.range(seq_length)
        position_ids = tf.broadcast_to(position_ids, input_ids.shape)
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)

        # shape: batch_size x seq_length x hidden_size
        word_embeddings = self.word_embeddings(input_ids)
        # shape: batch_size x seq_length x hidden_size
        position_embeddings = self.position_embeddings(position_ids)
        # shape: batch_size x seq_length x hidden_size
        segment_embeddings = self.segment_embeddings(segment_ids)

        # sum over 3 types of embedding, follow by layernorm and dropout layer
        embeddings = word_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # shape: batch_size x seq_length x hidden_size
        return embeddings

class BERTSelfAttention(Model):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size / config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Dense(self.all_head_size, input_shape=(config.hidden_size,), kernel_initializer=create_initializer(config.initializer_range))
        self.key = Dense(self.all_head_size, input_shape=(config.hidden_size,), kernel_initializer=create_initializer(config.initializer_range))
        self.value = Dense(self.all_head_size, input_shape=(config.hidden_size,), kernel_initializer=create_initializer(config.initializer_range))

        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
            Args:
                x: a tensor with shape batch_size x seq_length x attention_head_size*num_attention_heads
        """

        output_tensor = tf.reshape(x, [x.shape[0], x.shape[1], self.num_attention_heads, self.attention_head_size])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])

        # shape: batch_size x num_attention_heads x seq_length x attention_head_size
        return output_tensor

    def call(self, hidden_states, attention_mask):
        """
            Args:
                hidden_states: this actually includes from_tensor and to_tensor
                but in SelfAttention from_tensor and to_tensor are identical.
                    shape: batch_size x seq_length x hidden_size
                => In inference phase, we need to add batch_size = 1 dimension
                to the input

                attention_mask: is precomputed for all layers in BERTModel forward() function.
                                This desired from input_mask in InputFeatures.
                                See the details in BERTModel().
                    shape: batch_size x 1 x seq_length x seq_length
        """
        
        
        # shape of mixed_query|key|value_layer:
        # batch_size x seq_length x attention_head_size*num_attention_heads
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # shape: batch_size x num_attention_heads x seq_lenth x attention_head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # shape: batch_size x num_attention_heads x seq_length x seq_length
        attention_scores = tf.matmul(query_layer, key_layer)
        attention_scores /= math.sqrt(self.attention_head_size)

        # Apply the attention mask
        attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        # shape: batch_size x num_attention_heads x seq_length x seq_length
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # shape: batch_size x num_attention_heads x seq_length x attention_head_size
        context_layer = tf.matmul(attention_probs, value_layer)
        # shape: batch_size x seq_length x num_attention_heads x attention_head_size
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        # shape: batch_size x seq_length x num_attention_heads*attention_head_size
        context_layer = tf.reshape(context_layer, [context_layer.shape[0], context_layer.shape[1], self.num_attention_heads * self.attention_head_size])
        return context_layer

class BERTSelfOutput(Model):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size, kernel_initializer=create_initializer(config.initializer_range))
        self.LayerNorm = LayerNormalization()
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        """
            Args:
                hidden_states: output of BERTSelfAttention
                    shape: batch_size x seq_length x num_attention_heads*attention_head_size
                input_tensor: output of BERTEmbedding, use for residual connection
                    shape: batch_size x seq_length x hidden_size
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # apply residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BERTAttention(Model):
    def __init__(self, config):
        super().__init__()
        self.selfattention = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def call(self, input_tensor, attention_mask):
        """
            Args:
                attention_mask: 1.0 for sequence, 0.0 for padding.
                input_tensor: output of BERTEmbedding.
        """
        self_output = self.selfattention(input_tensor, attention_mask)
        attention_output = self.output(self_output)

        # shape: batch_size x seq_length x hidden_size
        return attention_output

class BERTIntermediate(Model):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.intermediate_size, input_shape=(config.hidden_size,), kernel_initializer=create_initializer(config.initializer_range))
        self.intermediate_act_fn = gelu

    def call(self, hidden_states):
        """
            Args:
                hidden_states: output of BERTAttention.
                    shape: batch_size x seq_length x hidden_size.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        # shape: batch_size x seq_length x intermediate_size
        return hidden_states

class BERTOutput(Model):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.hidden_size, input_shape=(config.intermediate_size,), kernel_initializer=create_initializer(config.initializer_range))
        self.LayerNorm = LayerNormalization()
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        """
            Args:
                hidden_states: output of BERTIntermediate.
                    shape: batch_size x seq_length x intermediate_size.
                input_tensor: output of BERTAttention.
                    shape: batch_size x seq_length x hidden_size.
                    use for residual connection
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # apply residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # shape: batch_size x seq_length x hidden_size
        return hidden_states

class BERTLayer(Model):
    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def call(self, input_tensor, attention_mask):
        """
            Args:
                input_tensor: output of BERTEmbedding layer.
                attention_mask: attention_mask: 1.0 for sequence, 0.0 for padding.
        """
        attention_output = self.attention(input_tensor, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        bert_layer_output = self.output(intermediate_output, attention_output)

        # shape: batch_size x seq_length x hidden_size
        return bert_layer_output

class BERTEncoder(Model):
    def __init__(self, config):
        super().__init__()
        bert_layer = BERTLayer(config)
        self.config = config
        self.layer = tf.keras.models.Sequential()
        for _ in range(config.num_hidden_layers):
            self.layer.add(bert_layer)

    def call(self, hidden_states, attention_mask):
        """
            Args:
                hidden_states: output of BERTEmbedding layer.
                    shape: batch_size x seq_length x hidden_size.
                attention_mask: attention_mask: attention_mask: 1.0 for sequence, 0.0 for padding.
                    shape: batch_size x seq_length x hidden_size.
        """

        for idx in range(self.config.num_hidden_layers):
            hidden_states = self.layer.get_layer(index=idx)(hidden_states, attention_mask)

        # shape: batch_size x seq_length x hidden_size
        return hidden_states

class BERTPooler(Model):
    def __init__(self, config):
        super().__init__()
        self.dense = Dense(config.hidden_size, input_shape=(config.hidden_size,), kernel_initializer=create_initializer(config.initializer_range))
        self.activation = tf.nn.tanh

    def call(self, hidden_states):
        """
            We "pool" the model by simply taking the hidden state corresponding
            to the first token.

            Args:
                hidden_states: output of BERTEncoder.
                    shape: batch_size x seq_length x hidden_size.
        """
        # shape: batch_size x seq_length x hidden_size
        cls_hidden_state = hidden_states
        cls_hidden_state = self.dense(cls_hidden_state)
        sequence_output = self.activation(cls_hidden_state)
        # shape: batch_size x hidden_size
        pooled_output = sequence_output[:, 0]

        # shape: batch_size x seq_length x hidden_size, batch_size x hidden_size
        return sequence_output, pooled_output

class BERTModel(Model):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embedding = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def call(self, input_ids, segment_ids=None, attention_mask=None):
        """
            Args:
                input_ids: each entry is idex of that word in vocabulary.
                    shape: batch_size x seq_length
                segment_ids: segment sentence A vs sentence B.
                    shape: batch_size x seq_length
                attention_mask: segment sentence A + B with padding.
                    shape: batch_size x seq_length
        """
        if attention_mask is None:
            attention_mask = tf.ones_like(input_ids)
        if segment_ids is None:
            segment_ids = tf.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length].
        # But BERTModel only uses self attention => from_seq_length = to_seq_length
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # with attention_mask of shape [batch_size, 1, 1, hidden_size].

        # This attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, axis=[1]), axis=[1])
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embedding(input_ids, segment_ids)
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        sequence_output, pooler_output = self.pooler(encoder_output)

        # shape: batch_size x seq_length x hidden_size, batch_size x hidden_size
        return sequence_output, pooler_output

class BertForSequenceClassification(Model):
    """
        BERT model for classification.
        This module is composed of the BERT model with a linear layer on top of
        the pooled output.

        Example usage:
        ```python
            # Already been converted into WordPiece token ids
            input_ids = tf.Tensor([[31, 51, 99], [15, 5, 0]])
            input_mask = tf.Tensor([[1, 1, 1], [1, 1, 0]])
            segment_ids = tf.Tensor([[0, 0, 1], [0, 2, 0]])

            config = BertConfig(vocab_size=32000, hidden_size=512,
                num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

            num_labels = 2

            model = BertForSequenceClassification(config, num_labels)
            probabilities, loss = model(input_ids, segment_ids, input_mask)
    """
    def __init__(self, config, num_labels):
        super().__init__()
        self.bert = BERTModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Dense(num_labels, input_shape=(config.hidden_size,), kernel_initializer=create_initializer(config.initializer_range))

    def call(self, input_ids, segment_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        # shape: batch_size x num_labels
        logits = self.classifier(pooled_output)

        # Calculate probabilities
        # shape: batch_size x num_labels
        probabilities = tf.nn.softmax(logits)


        if labels is not None:
            # Calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
            return loss, probabilities
        else:
            return probabilities

if __name__ == "__main__":
    bert_config = BertConfig.from_json_file("D:\ABSA\ABSA-BERT-pair-test\multi_cased_L-12_H-768_A-12\\bert_config.json")
    model = BertForSequenceClassification(bert_config, 4)
    model.summary()