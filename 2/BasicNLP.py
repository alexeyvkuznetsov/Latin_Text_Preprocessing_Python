import nltk
from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
from collections import Counter
from IPython.display import Image
from cltk.stop.latin import STOPS_LIST

# See http://docs.cltk.org/en/latest/latin.html#sentence-tokenization

cato_agri_praef = "Est interdum praestare mercaturis rem quaerere, nisi tam periculosum sit, et item foenerari, si tam honestum. Maiores nostri sic habuerunt et ita in legibus posiverunt: furem dupli condemnari, foeneratorem quadrupli. Quanto peiorem civem existimarint foeneratorem quam furem, hinc licet existimare. Et virum bonum quom laudabant, ita laudabant: bonum agricolam bonumque colonum; amplissime laudari existimabatur qui ita laudabatur. Mercatorem autem strenuum studiosumque rei quaerendae existimo, verum, ut supra dixi, periculosum et calamitosum. At ex agricolis et viri fortissimi et milites strenuissimi gignuntur, maximeque pius quaestus stabilissimusque consequitur minimeque invidiosus, minimeque male cogitantes sunt qui in eo studio occupati sunt. Nunc, ut ad rem redeam, quod promisi institutum principium hoc erit."
cato_agri_praef_lowered = cato_agri_praef.lower()
# create a tokenizer instance of the TokenizeSentence Class
latin_sentence_tokenizer = TokenizeSentence('latin')

#tokenize the text into sentence tokens
cato_sentence_tokens = latin_sentence_tokenizer.tokenize_sentences(cato_agri_praef)

# tokenize the text (or specific sentences) into specific words
latin_word_tokenizer = WordTokenizer('latin')
cato_word_tokens = latin_word_tokenizer.tokenize(cato_agri_praef_lowered)
cato_word_tokens_WO_punt = [token for token in cato_word_tokens if token not in ['.', ',', ':', ';']]

#print the tokens and the number of tokens
num_of_sentences = len(cato_sentence_tokens)
num_of_words = len(cato_word_tokens_WO_punt)
#print("There are " + str(num_of_sentences) + " sentences in the text")
#print("There are " + str(num_of_words) + " words in the text")
# for sentence in cato_sentence_tokens:
#     print(sentence)
#     print()

#print(cato_word_tokens_WO_punt)

#You can actually make the words unique by using a set
cato_word_tokens_WO_punt_unique = set(cato_word_tokens_WO_punt)
num_of_unique_words = len(cato_word_tokens_WO_punt_unique)
print("There are " + str(num_of_unique_words) + " unique words in the text")
print(cato_word_tokens_WO_punt_unique)

#lets alphabetize all the words in the list

alphabetized_list = []

for word in cato_word_tokens_WO_punt_unique:
    alphabetized_list.append(word)

alphabetized_list.sort()

#print the new list

print(alphabetized_list)

#Count the amount of times that words occur
#The counter is actually a dictionary, so you can look up
#The frequency of specific words
cato_word_counts_counter = Counter(cato_word_tokens_WO_punt)
print(cato_word_counts_counter)


#todo go back and figure out how to show charts and graphs

Image('images/tableau_bubble.png')

#these are all the stopwords included in the stops list for latin
print(STOPS_LIST)

cato_no_stops = [w for w in cato_word_tokens_WO_punt if not w in STOPS_LIST]
print(cato_no_stops)

cato_no_stops_counter = Counter(cato_no_stops)
print(cato_no_stops_counter)
