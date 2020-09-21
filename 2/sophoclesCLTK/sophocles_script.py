from cltk.stem.lemma import LemmaReplacer
from cltk.stop.greek.stops import STOPS_LIST

file1 = open("antigone.txt")
antigone = file1.read()
file1.close()

file2 = open("odeToMan.txt")
odeToMan = file2.read()
file2.close()

def lemmata(text):
    lemmatizer = LemmaReplacer('greek')
    return [word for word in set(lemmatizer.lemmatize(text.lower())) if not word in STOPS_LIST]

antigoneLemmata = lemmata(antigone)
odeLemmata = lemmata(odeToMan)

sharedLemmata = set(antigoneLemmata).intersection(odeLemmata)

numAntigoneLemmata = len(antigoneLemmata)
numOdeLemmata = len(odeLemmata)
numSharedLemmata = len(sharedLemmata)

print(numAntigoneLemmata, numOdeLemmata, numSharedLemmata)
