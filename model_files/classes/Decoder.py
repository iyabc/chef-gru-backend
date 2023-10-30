import tensorflow as tf
from model_files.classes.CrossAttention import CrossAttention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, units, decodedTokenizer):
        super(Decoder, self).__init__()
        self.decodedTokenizer = decodedTokenizer
        self.vocab_size = decodedTokenizer.get_vocab_size()
        self.start_token = decodedTokenizer.encode("[START]").ids
        self.end_token = decodedTokenizer.encode("[END]").ids
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True
        )

        self.rnn = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        self.attention = CrossAttention(units)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        logits = self.output_layer(x)

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.decodedTokenizer.decode_batch(tokens.numpy())
        result = tf.strings.regex_replace(words, "^ *\[START\] *", "")
        result = tf.strings.regex_replace(result, " *\[END\] *$", "")
        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        logits, state = self(context, next_token, state=state, return_state=True)

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        done = done | (next_token == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state
