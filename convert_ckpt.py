import tensorflow as tf
from modelling import BertConfig, BERTModel
import numpy as np
import os

reader = tf.train.load_checkpoint("/home/ddkhai/Documents/ABSA/uncased_L-12_H-768_A-12/bert_model.ckpt")

bert_config = BertConfig.from_json_file("/home/ddkhai/Documents/ABSA/uncased_L-12_H-768_A-12/bert_config.json")
model = BERTModel(bert_config)

# Test new checkpoint
# restore_ckpt = tf.train.Checkpoint(model=model)
# status = restore_ckpt.restore("/home/ddkhai/Documents/ABSA/bert_tensorflow/new_checkpoint/bert_model.ckpt")

f_input_ids = np.zeros((3, 100))
f_segment_ids = np.zeros((3, 100))
f_attention_mask = np.zeros((3, 100))

model(f_input_ids, f_segment_ids, f_attention_mask)

for v in model.variables:
    name = v.name
    name = name.replace("bert_model/", "bert/")
    name = name.replace("/embeddings:0", "")
    name = name.replace(":0", "")

    ckpt_tensor = reader.get_tensor(name)
    tf.compat.v1.assign(v, ckpt_tensor, True)

ckpt = tf.train.Checkpoint(model=model)
if not os.path.exists("./new_checkpoint"):
    os.mkdir("./new_checkpoint")
ckpt.save("./new_checkpoint/bert_model.ckpt")