
from cltk.corpus.utils.importer import CorpusImporter
from cltk.tokenize.word import WordTokenizer
from collections import Counter
from cltk.stop.latin import STOPS_LIST
from cltk.stem.lemma import LemmaReplacer
from cltk.corpus.latin import corpora
from cltk.corpus.readers import get_corpus_reader
from src.cantusNlp.utils.Lemma import LEMMATA


class CltkOperator:

    def __init__(self):
        print("")

    def retrieveLatinModels(self):
        """
        Loads the required Latin data models (for the cltk processing) from the internet.
        Uses the CorpusImporter('latin') to access the resources.
        The data will be stored in the local project ...> from then the cltk
        """
        latinDownloader = CorpusImporter('latin')
        latinDownloader.import_corpus('latin_text_latin_library')
        latinDownloader.import_corpus('latin_models_cltk')

    def wtokenizeLatin(self, text: str, removeSplitSyllable: bool = False) -> ([str]) or ([str], [str]):
        """
        Uses the latin word tokenizer from cltk to tokenize the words for given text.
        Removes punctuation internally.
        :param text: Text to tokenize.
        :param removeSplitSyllable: true ..> when "big" words are split by the cltk tokenizer
        it adds split syllabi (like "-que") to the return array. If this param is set to true these
        split syllabi are being removed otherwise not.
        :return: Tuple with List Comprehension of tokenized words on first index position. When
        paramater removeSplitSyllable was assigned true ...> returns on second index position the
        removed words (also as list comprehension).
        """
        text = text.lower()
        wordTokenizer = WordTokenizer("latin")
        tokens: [] = wordTokenizer.tokenize(text)

        return_tuple: tuple = ()
        removed_words: [] = []
        if removeSplitSyllable:
            for word in tokens:
                if "-" in word:
                    removed_words.append(word)
                    tokens.remove(word)

            return_tuple = (tokens, removed_words)
            return return_tuple

        return_tuple = (tokens)
        return return_tuple

    def countWords(self, text: list):
        wordsDict: dict = Counter(text)
        return wordsDict

    def removeLatStopWords(self, tokenized_text: list):
        removedStops: list[str] = [w for w in tokenized_text if not w in STOPS_LIST]
        return removedStops

    def lemmatizeLat(self, tokenized_words: list, return_raw: bool = False) -> ([str]) or ([str], [str]):
        """
        Lemmatizes given list of words against the cltk perseus corpus. If second parameter
        is set to true -> returns a list with words BUT additionally with derived "source-word" after
        a "/" seperator.
        :param tokenized_words: String list of words to be lemmatized.
        :param return_raw: Boolean, decides if return should contain raw "source word" or not.
        :return: First index position -> List of lemmmatas; Second index position -_> if second parameter was true
        then list of lemmatas with "source_words" attached to each lemmata string BEFORE the lemmatized word.
        """
        lemmatizer = LemmaReplacer('latin')
        lemmata: [str] = lemmatizer.lemmatize(tokenized_words, False)
        if return_raw:
            lemmata_with_source: [str] = lemmatizer.lemmatize(tokenized_words, True)
            return lemmata, lemmata_with_source
        else:
            return lemmata

    def analyzeCltkLemmaDeviation(self, wordList: list) -> ([str], float):
        """
        Takes in a wordList and compares each word wo be inside the lemmatalist of the cltk (retrieved from perseus corpus).
        Returns words not matched as list comprahension
        :param wordList:
        :return: Tuple: on first index position a list comprehension of all words that were not
        matched from the lemma-list (perseus corpus), on second index position: percentage of words
        that could not be found as FLOAT.
        """
        lemma_array = []
        for key in LEMMATA:
            key = str.lower(key)
            lemma_array.append(key)

        words_not_inside_lemma: [str] = []
        words_found_counter: int = 0
        words_not_found_counter: int = 0
        for word in wordList:
            if word not in lemma_array:
                words_not_inside_lemma.append(word)
                words_not_found_counter += 1
            else:
                words_found_counter += 1

        words_total = words_found_counter + words_not_found_counter
        percentage_no_match: float = (100/words_total)*words_not_found_counter

        return words_not_inside_lemma, percentage_no_match
