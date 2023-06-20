from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from nltk.corpus import wordnet
import os
import spacy
import re
import random


class Shuffler():
    """
    This class is used to apply shuffling to observations.
    """
    def __init__ (self):
        self.nlp = spacy.load("en_core_web_sm")

    def generate(self, input):
        """
        Shuffle tokens in `input`.

        Args:
            input (list[str]) : Batch input sentences
        
        Returns:
            shuffled (list[str]) : The shuffled observations.
        """ 

        docs = self.nlp.pipe(input)

        sents = [[t.text for t in d] for d in docs]
        for sent in sents:
            random.shuffle(sent)
        shuffled = [" ".join(s) for s in sents]
        return shuffled


class Simplifier():
    """
    This class applies simplification to observations, such as filtering only certain POS tags.
    """

    def __init__(self, pos_filter={"NOUN", "PROPN", "ADJ", "VERB"}):
        self.pos_filter = pos_filter
        self.nlp = spacy.load("en_core_web_sm")

    def generate(self, inputs):
        """
        Generates simplified observations from `inputs`.

        Args:
            inputs(list[str]) : Batched inputs
        
        Returns:
            simp_obs (list[str]) : Simplified observations
        """

        docs = self.nlp.pipe(inputs)
        sents = [[t.text for t in d if t.pos_ in self.pos_filter] for d in docs]
        simp_obs = [" ".join(s) for s in sents]
        return simp_obs


class Summarizer():
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
    
    def summarize(self, input):
        prefix = 'summarize: '
        input = [prefix + t for t in input]
        model_inputs = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model.generate(input_ids=model_inputs['input_ids'],
                                      attention_mask=model_inputs['attention_mask'],
                                      do_sample=False)

        return self.decode(outputs)
    
    def generate(self, text):
        summ = self.summarize(text)

        return summ


    def decode(self, token_id_sequences):
        """
        Batch decode token id sequences.

        Args:
            token_id_sequences(torch.Tensor) : n x L tensor with token id sequences.
        
        Returns:
            texts(list) : The list of decoded strings.
        """
        return self.tokenizer.batch_decode(token_id_sequences, skip_special_tokens=True)


class Paraphraser():

    def __init__(self, language='de'):

        if os.path.exists("/dccstor/mgruppi1/"):
            torch.hub.set_dir('/dccstor/mgruppi1/cache/')

        if language == 'fr':
            self.from_en = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')
            self.to_en = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.fr-en', tokenizer='moses', bpe='subword_nmt')
        elif language == 'de':
            self.from_en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.to_en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
        self.from_en.cuda()
        self.to_en.cuda()

        self.nlp = spacy.load("en_core_web_sm")

        self.cache = dict()

    def translate_two_way(self, text, n=5):
        """
        Translate `text` in both ways.
        
        Args:
            text(str) : The text to be paraphrased.
            n(int) : Number of results to output.
        
        Returns:
            output(list) : List of translations.
        """
        if text not in self.cache:
            output = list()
            text = text.strip()
            translated = self.from_en.translate(text)
            t_tokens = self.to_en.tokenize(translated)
            t_bpe = self.to_en.apply_bpe(t_tokens)
            t_bin = self.to_en.binarize(t_bpe)

            en_bin = self.to_en.generate(t_bin, beam=n+1, nbest=n+1, topk=20)

            for i in range(n):
                en_sample = en_bin[i]['tokens']
                en_bpe = self.to_en.string(en_sample)
                en_toks = self.to_en.remove_bpe(en_bpe)
                en = self.to_en.detokenize(en_toks)
                output.append(en)

            self.cache[text] = output
        else:
            output = self.cache[text]

        return output

    def generate(self, text, n=5, sentencize=False):
        """
        Generate `n` paraphrasings for the input text.
        
        Args:
            text(str) : The input text.
            n(int) : The number of paraphrasings to generate.
            sentencize(bool) : If True, apply paraphrasing to each sentence, individually.
        
        Returns:
            output(list) : The list of `n` paraphrasings for the input text. 
        """

        if not sentencize:
            output = self.translate_two_way(text, n)
        else:
            output = [list() for _ in range(n)]

            # Sentencize input
            doc = self.nlp(text)

            for sent in doc.sents:
                paraphrasings = self.translate_two_way(sent.text)
                for i in range(n):
                    output[i].append(paraphrasings[i])
            
            # Once done, join all sentences in each `output`.
            for i in range(n):
                output[i] = " ".join(output[i])

        return output


class Synset:
    def __init__(self, entities={}):
        self.wordnet = wordnet
        self.nlp = spacy.load("en_core_web_sm")

        # # For n-gram entities, we extract the root of the noun chunk
        # noun_ents = list()
        # for e in entities:
        #     nc_root = ''
        #     for nc in self.nlp(e).noun_chunks:
        #         nc_root = nc.root.text
        #     noun_ents.append(nc_root)
        # entities = noun_ents

        # self.entities = entities  # Entities to be replaced in observations (e.g. the name of game objects)
        
        # # Init hypernyms, synonyms and replacements
        # self.hypernyms = {e: self.get_hypernyms(e)[0] for e in self.entities}
        # self.synonyms = {e: self.get_synonyms(e)[0] for e in self.entities}
        # self.replacements = {e: self.get_replacement(e)[0] for e in self.entities}

    def get_synsets(self, word):
        return self.wordnet.synsets(word)
    
    def get_first_synset(self, word):
        return self.wordnet.synsets(word)[0]
    
    def get_synonyms(self, word, synset=0):
        """
        Returns a list of synonyms drawn from the first synset (or any arbitrary one, if chosen).

        Args:
            word(str) : The word to get a synonym for.
            synset(int) : The synset to use. Default=0, the first (most common) synset.
        
        Returns:
            synonym(list[str]) : The list of synonyms found for `word`.
        """

        synsets = self.wordnet.synsets(word)
        if synset >= len(synsets):
            # print("No synset for", word)
            return [word]

        syn_lemmas = synsets[synset].lemmas()
        synonyms = [s.name() for s in syn_lemmas if s.name().lower() != word.lower()]
        
        if len(synonyms) == 0:
            synonyms = [word]

        return synonyms
    
    def get_hypernyms(self, word, synset=0):
        """
        Returns the hypernyms of `word`.

        Args:
            word(str) : The input word.
            synset(int) : The synset to get the hypernym from.

        Returns:
            hypernyms(list[str]) : The list of hypernyms.
        """
        synsets = self.wordnet.synsets(word)
        if synset >= len(synsets):
            # print("No synset for", word)
            return [word]

        hyps = synsets[synset].hypernyms()
        hyp_lemmas = list()
        for h in hyps:
            hyp_lemmas.extend([l.name() for l in h.lemmas()])
        
        if len(hyp_lemmas) == 0:
            hyp_lemmas = [word]

        return hyp_lemmas
    
    def get_replacement(self, word, synset=0):
        """
        Get a word replacement by sampling a random hyponym of the word's hypernym.
        E.g.: if `word` is 'apple' and its hypernym is 'fruit', potential replacements could be 'orange' or 'banana'.

        Args:
            word(str) : The word to be replaced.
            synset(int) : The number of the synset to use.

        Returns:
            replacements(list[str]) : The list of hypernym's hyponyms.
        """

        synsets = self.wordnet.synsets(word)
        if synset >= len(synsets):
            # print("No synset for", word)
            return [word]
        hypernyms = synsets[synset].hypernyms()
        if len(hypernyms) == 0:
            # Cannot get a hypernym
            return [word]
        hyponyms = hypernyms[0].hyponyms()
        if len(hyponyms) == 0:
            # No hyponyms found
            return [word]
        lemmas = hyponyms[0].lemma_names()

        return lemmas

    def generate(self, inputs, how="replacement"):
        """
        Generate lexical substitute for inputs

        Args:
            inputs (list[str]) : Batch inputs
        
        Returns:
            outputs (list[str)]) : Output with substitutes
        """

        # if how == "synonym":
        #     targets = self.synonyms
        # elif how == "hypernym":
        #     targets = self.hypernyms
        # elif how == "replacement":
        #     targets = self.replacements
        if how == "synonym":
            repl = self.get_synonyms
        elif how == "hypernym":
            repl = self.get_hypernyms
        elif how == "replacement":
            repl = self.get_replacement

        outputs = list()
        docs = self.nlp.pipe(inputs)
        for doc in docs:
            new_text = list()
            for token in doc:
                if token.pos_ == "NOUN":
                    # tk = targets[token.text] + token.whitespace_
                    tk = repl(token.text)[0] + token.whitespace_
                else:
                    tk = token.text_with_ws
                new_text.append(tk)
            text = "".join(new_text)
            outputs.append(text)
        # outputs = [self.substitute_entities(text) for text in inputs]
        return outputs


    def substitute_entities(self, text, entities, target="replacement"):
        """
        Substitute in the input `text` the words in `entities` with some criterion such as synonyms or hypernyms.

        Args:
            text(str) : The input text.
            entities(list[str]) : The list of entities to be substituted.
            target(str) : The type of target word to use as replacement ('synonym' or 'hypernym' or 'replacement').
        
        Returns:
            text_sub(str) : The text with all entities substituted.
        """

        # For n-gram entities, we extract the root of the noun chunk
        noun_ents = list()
        for e in entities:
            nc_root = ''
            for nc in self.nlp(e).noun_chunks:
                nc_root = nc.root.text
            noun_ents.append(nc_root)
        entities = noun_ents

        if target == 'synonym':
            targets = {e: self.get_synonyms(e)[0] for e in entities}  # Get one synonym per entity.
        elif target == 'hypernym':
            targets = {e: self.get_hypernyms(e)[0] for e in entities}  # Get one hypernym per entity.
        elif target == 'replacement':
            # In this mode, we get the word's hypernym and a random hyponym
            targets = {e: self.get_replacement(e)[0] for e in entities} # Get replacement entity.

        print("Subs:", targets)

        doc = self.nlp(text)
        new_text = list()
        for token in doc:
            if token.text in targets:
                tk = targets[token.text] + token.whitespace_
            else:
                tk = token.text_with_ws
            new_text.append(tk)
        text = "".join(new_text)
        
        return text        


if __name__ == "__main__":
    # Test

    inputs = ["This is a sentence in English.", "I like cheese very much"]

    # summa = Summarizer()
    # print("Inputs", inputs)
    # print("-- Summarizing", summa.generate(inputs))

    # paraph = Paraphraser()
    # for txt in inputs:
    #     print(txt)
    #     print("-- Paraphrasing", *paraph.generate(txt, n=10), sep='\n - ')

    synseter = Synset()

    word = "sneakers"

    syns= synseter.get_synonyms(word)
    print("Synonym", *syns)

    hyps = synseter.get_hypernyms(word)
    print("Hypernyms", *hyps)

    shuffler = Shuffler()
    simplifier = Simplifier()

    print("SHUFFLED", shuffler.generate(inputs))
    print("SIMPLIFIED", simplifier.generate(inputs))
