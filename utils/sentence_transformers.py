from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling.
    The sentence encoding is given by the average of the token embeddings 
    in the sentence, after applying the attention mask.
    The input `model_output` is a tensor containing the token embeddings
    of a transformer model.
    Arguments:
        model_output: torch.Tensor -- The token embedding tensor
        attention_mask: torch.Tensor -- the attention mask of the input
    Returns:
        sentence_embeddings: torch.Tensor -- the fixed-length encodings of the input
    """
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def weighted_mean_pooling(model_output, attention_mask, attention_weights):
    """
    Performs weighted mean pooling of the input.
    All the inputs must have the same dimensions.
    The mean pooling is weighted by the weights in `attention_weights`.
    Arguments:
        model_output: torch.Tensor -- token embeddings
        attention_mask: torch.Tensor -- binary attention mask
        attention_weights: torch.Tensor -- attention weights to use
    """
    token_embeddings = model_output[0]  # Get token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    input_attention_expanded = attention_weights.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded * input_attention_expanded, 1) / torch.clamp(input_attention_expanded.sum(1), min=1e-9)


def normalize(x):
    """
    Returns row-normalized version of `x` using Frobenius norm.
    """
    return x/np.linalg.norm(x, axis=1, keepdims=True)


class Tokenizer():
    def __init__(self, model_name):
        self.model = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self, batch):
        return self.model(batch, padding=True, truncation=True, return_tensors='pt')
        

class Encoder():
    def __init__(self, model_name, device, compute_gradients=False):
        """
        
        Args:
            model_name (str) : Name of the Huggingface model.
            device (str) : CUDA device or CPU.
            compute_gradient (bool) : If False, model is put in eval mode.
        """
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        if not compute_gradients:
            self.model.eval()
    
    def encode(self, encoding_batch, no_grad=True):
        model_output = self.model(**encoding_batch)
        return mean_pooling(model_output, encoding_batch['attention_mask'])


class SentenceEncoder():
    def __init__(self, model_name='sentence-transformers/all-distilroberta-v1',
                 output_attentions=True,
                 device="cuda:0",
                 default_pooling='mean'):
        """
        Contruct a SentenceEncoder object.
        Arguments:
            model_name: str -- The name of the model to download from HuggingFace
            output_attentions: bool -- If `True`, returns the model attentions
            device: str -- CUDA device to use.
            default_pooling(str) : The default pooling strategy to use.
        """
        # Load model from HuggingFace Hub
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=output_attentions)
        self.device=device
        self.model.to(device)
        self.default_pooling = default_pooling

    def encode_input(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, 
                                        truncation=True, return_tensors='pt')
        return encoded_input
    
    def __call__(self, sentences, **kwargs):
        """
        Shorthand for `self.encode`.
        Arguments:
            sentences: list -- List of sentences as strings (not tokenized or indexed)
            **kwargs: dict -- `**kwargs` to send to `self.encode`.
        Returns:
            sentence_embeddings: torch.Tensor -- `n` x `d` embedding tensor
            attention: np.ndarray -- `n` x `m` attention weights for the given token 
        """

        return self.encode(sentences, **kwargs)

    def encode(self, sentences, attention_layer=-1, attention_idx=0,
                zero_attention_idx=False, return_attention_weights=False,
                return_tokens=False, return_encodings=False, no_grad=True,
                pooling=None):
        """
        Encodes a list of sentences using the currently loaded model.
        Attention is returned by selecting the layer from which retrieve the attention
        using `attention_layer`, and by averaging the weights of all attention heads
        in that layer for a given token defined by `attention_idx`, by default, the [CLS] token.
        The attention is returned as a `n x m` matrix where `n` is the number of sentences and `m`
        is the maximum sentence length.
        Arguments:
            sentences: list -- List of `n` sentences as strings
            attention_layer: int -- Which attention layer to pool from
            attention_idx: int -- index to get attention for. Default is 0, which implies obtaining the attention for the [CLS] token.
            zero_attention_idx: bool -- If True, the attention weights for `attention_idx` are zeroed out and the vector is re-normalized
            return_attention_weights: bool -- If True, attention weights are returned along with embeddings.
            return_tokens(bool) : If True, return the tokens extracted with the tokenizer.
            return_encodings(bool) : If True, return the encodings from the tokenizer.
            no_grad(bool) : If True, do not perform gradient calculation for embeddings (inference only).
            pooling: str -- Method of pooling to use, one of {'mean', 'weighted_mean'}. Default is `None`, which uses the default pooling strategy.
        Returns:
            sentence_embeddings: torch.Tensor -- `n` x `d` embedding tensor
            attention: np.ndarray -- `n` x `m` attention weights for the given token (if `return_attention_weights`)
        """
        encoded_input = self.encode_input(sentences).to(self.device)
        # Compute token embeddings
        if no_grad:
            with torch.no_grad():
                model_output = self.model(**encoded_input)
        else:
            model_output = self.model(**encoded_input)
        
        if pooling is None:
            pooling = self.default_pooling

        # If output_attentions is true, then:
        #  model_output[0] contains the token embeddings
        #  model_output[2] contains the token attentions
        #  If there are `l` layers and `k` attention heads in the model,
        #  then model_output[2] is shaped as (`n`, `l`, `k`, `p`, `p`)
        #  where `p` is the number of words + paddings in the input.
        # After encoding, sentences start with the [CLS] token, (id=101) and
        # end with the [SEP] token (id=102) with possibly trailing [PAD] tokens (id=0)

        if return_attention_weights or pooling == 'weighted_mean':
            h_attention = model_output[2][attention_layer]
            attention_weights = h_attention.mean(dim=1)[:, attention_idx]

            if zero_attention_idx:
                attention_weights = attention_weights.numpy()
                attention_weights[:, attention_idx] = 0
                attention_weights = normalize(attention_weights)  # normalize rows of `attention`

        # Perform pooling
        if pooling == 'mean':
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        elif pooling == 'weighted_mean':  # Can ONLY be used if self.output_attentions==True
            sentence_embeddings = weighted_mean_pooling(model_output, encoded_input['attention_mask'],
                                                     attention_weights)
        else:
            raise NotImplementedError("Pooling method '%s' not implemented." % pooling)

        output = {'embeddings': sentence_embeddings }
        if return_attention_weights:
            output['attention_weights'] = attention_weights
        if return_encodings:
            output['encodings'] = encoded_input
        if return_tokens:
            output['tokens'] = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])

        return output


if __name__ == "__main__":
    sentences = [
                "You see a table. There is an apple on the table.",
                "You see an apple over a table in the corner of the room.",
                "You see a table. There is an orange on the table.",
                "You make out something on the table. It looks like an apple.",
                "You see a table with a hat on it.",
                "You see a table with an orange on it.",
                "You see a table with an apple on it.",
                "You see a table but there is not a thing on it."
                ]
    batched_sequences = [ sentences, sentences, sentences]

    tokenizer = Tokenizer()
    encoder = Encoder()

    encodings = tokenizer.encode(sentences)
    embeddings = encoder.encode(encodings)

    print(embeddings.shape)

    # params = {
    #     "output_attentions": True,
    #     'device': 'cuda:0',
    # }
    # encoder = SentenceEncoder(**params)
    # encodings = encoder.encode(sentences, return_attention_weights=True,
    #                             no_grad=False, pooling='weighted_mean')

    # embeddings = encodings['embeddings']
    # print(embeddings)
    # attentions = encodings['attention_weights']

    # print("Single sequence", embeddings.shape)
