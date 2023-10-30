from tokenizers import ByteLevelBPETokenizer
from model_files.classes import constants as c


def getChefTokenizer(
    encodedPathVocab, encodedPathMerges, decodedPathVocab, decodedPathMerges
):
    encodedTokenizer = ByteLevelBPETokenizer(
        encodedPathVocab,
        encodedPathMerges,
    )

    encodedTokenizer.add_tokens(c.NEW_TOKENS)

    decodedTokenizer = ByteLevelBPETokenizer(
        decodedPathVocab,
        decodedPathMerges,
    )

    decodedTokenizer.add_tokens(c.NEW_TOKENS)

    return encodedTokenizer, decodedTokenizer
