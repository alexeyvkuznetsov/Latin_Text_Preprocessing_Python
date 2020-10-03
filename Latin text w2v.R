########################################################################
########################################################################
##  Latin Texts Word2vec representation                                         ##
##  Author: Alexey Kuznetsov                                          ##
##  URL: https://github.com/alexeyvkuznetsov/Latin_Text_Preprocessing ##
##       https://alexeyvkuznetsov.github.io                           ##
########################################################################
########################################################################
library(tm)
library(udpipe)
library(textmineR)

setwd("C:/GitHub/XML/")


prologus<-paste(scan(file ="files/01 prologus.txt",what='character'),collapse=" ")
historia_g<-paste(scan(file ="files/02 historia_g.txt",what='character'),collapse=" ")
recapitulatio<-paste(scan(file ="files/03 recapitulatio.txt",what='character'),collapse=" ")
historia_w<-paste(scan(file ="files/04 historia_w.txt",what='character'),collapse=" ")
historia_s<-paste(scan(file ="files/05 historia_s.txt",what='character'),collapse=" ")

prologus<-data.frame(texts=prologus)
historia_g<-data.frame(texts=historia_g)
recapitulatio<-data.frame(texts=recapitulatio)
historia_w<-data.frame(texts=historia_w)
historia_s<-data.frame(texts=historia_s)

prologus$book<-"1 Prologus"
historia_g$book<-"2 Historia Gothorum"
recapitulatio$book<-"3 Recapitulatio"
historia_w$book<-"4 Historia Wandalorum"
historia_s$book<-"5 Historia Suevorum"

historia<-rbind(prologus,historia_g,recapitulatio,historia_w,historia_s)

#historia$texts <- stripWhitespace(historia$texts)
historia$texts <- tolower(historia$texts)
historia$texts <- removePunctuation(historia$texts)
historia$texts <- removeNumbers(historia$texts)

# Stopwords

load("rome_number_1000.Rda")

customStopWords <- c("ann", "annus", "aer", "aes", "aera", "num._rom.", "xnum._rom.", "xxnum._rom.", "xxxnum._rom.", "cdxlnum._rom.")

lat_stop_perseus <- c("ab", "ac", "ad", "adhic", "aliqui", "aliquis", "an", "ante", "apud", "at", "atque", "aut", "autem", "cum", "cur", "de", "deinde", "dum", "ego", "enim", "ergo", "es", "est", "et", "etiam", "etsi", "ex", "fio", "haud", "hic", "iam", "idem", "igitur", "ille", "in", "infra", "inter", "interim", "ipse", "is", "ita", "magis", "modo", "mox", "nam", "ne", "nec", "necque", "neque", "nisi", "non", "nos", "o", "ob", "per", "possum", "post", "pro", "quae", "quam", "quare", "qui", "quia", "quicumque", "quidem", "quilibet", "quis", "quisnam", "quisquam", "quisque", "quisquis", "quo", "quoniam", "sed", "si", "sic", "sive", "sub", "sui", "sum", "super", "suus", "tam", "tamen", "trans", "tu", "tum", "ubi", "uel", "uero", "unus", "ut", "quoque", "xiix")

#save(lat_stop_perseus,file="lat_stop_perseus.Rda")

#load("lat_stop_perseus.Rda")

#MyStopwords <- c(lat_stop_perseus, customStopWords, lat_stopwords_romnum)

MyStopwords <- c(lat_stop_perseus, rome_number_1000, customStopWords)

#historia$texts <- removeWords(historia$texts, c(lat_stop_perseus, rome_number_1000))

historia$texts <- removeWords(historia$texts, MyStopwords)

historia$texts <- stripWhitespace(historia$texts)



udmodel_latin <- udpipe_load_model(file = "latin-ittb-ud-2.5-191206.udpipe")


x <- udpipe_annotate(udmodel_latin, x = historia$texts, doc_id = historia$book, tagger = "default", parser = "default", trace = TRUE)
x <- as.data.frame(x)

x <- paste.data.frame(x, term = "lemma", group = "doc_id", collapse = " ")


library(word2vec)
set.seed(123456789)
model <- word2vec(x = x$lemma, dim = 15, iter = 20, split = c(" ", ".\n?!"))
embedding <- as.matrix(model)


#write.word2vec(model, "isidoremodel.bin")
model <- read.word2vec("isidoremodel.bin")
terms <- summary(model, "vocabulary")
embedding <- as.matrix(model)


embedding <- predict(model, c("gens", "populus"), type = "embedding")
lookslike <- predict(model, "gens", type = "nearest", top_n = 10)
lookslike




#library(devtools)
#install_github("mukul13/rword2vec")

library(rword2vec)
ls("package:rword2vec")
textus = x$lemma
model=word2vec(train_file = "textus.txt",output_file = "vec.bin",binary=1)
dist=distance(file_name = "vec.bin",search_word = "gens",num = 20)


dist=distance(file_name = "vec.bin",search_word = "gens",num = 20)


write(textus, "textus.txt")





library(wordVectors)



model = train_word2vec("textus.txt","textus.bin",vectors=200,threads=4,window=12,iter=5,negative_samples=0)

# Чтение модели
model <- read.vectors("isidoremodel.bin")

model %>% closest_to("gens")




















dtf <- subset(x, upos %in% c("NOUN"))

dtf <- document_term_frequencies(dtf, document = "doc_id", term = "lemma")

head(dtf)


## Create a document-term matrix
dtm <- document_term_matrix(x = dtf)

## Remove words which do not occur that much
dtm <- dtm_remove_lowfreq(dtm, minfreq = 2)

head(dtm_colsums(dtm))

## Remove stopwords
dtm <- dtm_remove_terms(dtm, terms = c("ann.", "ann", "an", "annus", "aer", "aes", "suus", "filius", "pater", "frater", "pars", "maldra", "theudericus", "gothus", "hucusque", "hispanium", "caeter", "justinianus", "praelio", "cdxxxnum._rom.", "cdxinum._rom.", "cdxix", "op"))

dtm <- dtm_remove_terms(dtm, terms = c("ann.", "ann", "an", "annus", "aer", "aes", "suus", "filius", "pater", "frater", "pars", "maldra", "theudericus", "hucusque", "hispanium", "caeter", "justinianus", "praelio", "cdxxxnum._rom.", "cdxinum._rom.", "cdxix", "op"))


# Create a term-document matrix
dtm <- as.matrix(dtm)
tdm <- t(dtm)

# Convert a DTM to a Character Vector of documents
library(textmineR)
dtm.to.docs <- textmineR::Dtm2Docs(dtm = dtm)


## Convert dtm to a list of text
dtm.to.docs <- apply(dtm, 1, function(x) {
  paste(rep(names(x), x), collapse=" ")
})

## convert list of text to a Corpus

myCorpus <- VCorpus(VectorSource(dtm.to.docs))
inspect(myCorpus)

# Created term-document matrix from corpus

tdm <- TermDocumentMatrix(myCorpus)

td_matrix <- as.matrix(tdm)


