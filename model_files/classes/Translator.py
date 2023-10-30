from model_files.classes.Encoder import Encoder
from model_files.classes.Decoder import Decoder
import tensorflow as tf


class Translator(tf.keras.Model):
    def __init__(self, units, encodedTokenizer, decodedTokenizer):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(units, encodedTokenizer)
        decoder = Decoder(units, decodedTokenizer)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        return logits

    def translate(self, texts, *, max_length=250, temperature=0.0):
        context = self.encoder.convert_input(texts)
        batch_size = 1

        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        for _ in range(max_length):
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done, state, temperature
            )

            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        tokens = tf.concat(tokens, axis=-1)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)

        result = self.decoder.tokens_to_text(tokens)
        return result
