from transformers import AutoTokenizer, AutoModel
from model_files import constants as c
from tokenizers import ByteLevelBPETokenizer


def getTokenizer():
    # model_name = "gpt2"
    # model_name = "bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tempNewTokens = c.NEW_TOKENS
    # tempNewTokens.append("<unknown>")
    # new_tokens = list(set(tempNewTokens) - set(tokenizer.get_vocab().keys()))
    # tokenizer.add_tokens(new_tokens)
    number_of_recipes = c.NUMBER_OF_RECIPES

    tokenizer = ByteLevelBPETokenizer(
        f"model_files/tokenizers/chef_tokenizer_{number_of_recipes}-vocab.json",
        f"model_files/tokenizers/chef_tokenizer_{number_of_recipes}-merges.txt",
    )

    tokenizer.add_tokens(c.NEW_TOKENS_NO_SPACES)

    return tokenizer