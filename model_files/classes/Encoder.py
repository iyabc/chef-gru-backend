import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, encodedTokenizer):
        super(Encoder, self).__init__()
        self.vocab_size = encodedTokenizer.get_vocab_size()
        self.units = units
        self.encodedTokenizer = encodedTokenizer
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True
        )

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="sum",
            layer=tf.keras.layers.GRU(
                units,
                return_sequences=True,
                recurrent_initializer="glorot_uniform",
            ),
        )

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        return x

    def convert_input(self, texts):
        ids = self.encodedTokenizer.encode(texts).ids
        ids_tensor = tf.convert_to_tensor(ids)[tf.newaxis]
        context = self(ids_tensor)
        return context
