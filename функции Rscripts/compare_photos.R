library(udpipe)
library(text2vec)
library(tm)
library(textreuse)
library(RWeka)
library(slam)
library(stringr)
czech=udpipe_load_model("czech-ud-2.0-170801.udpipe")
compare_docs_fotka_two = function(source,target,n){
  my_tokenize = function(x) NGramTokenizer(x, Weka_control(min = n, max = n))
  docs<-VCorpus(VectorSource(c(source,target)))
  tdm <- TermDocumentMatrix(docs, control = list(tokenize = my_tokenize))
  cosine_dist_mat <- 1 - crossprod_simple_triplet_matrix(tdm)/(sqrt(col_sums(tdm^2) %*% t(col_sums(tdm^2))))
  cosine_dist_mat
}


lemmatize = function(docs){
  res = c()
  for(i in docs){
    ann = as.data.frame(udpipe_annotate(czech,i))
    res=c(res,paste(ann$lemma,collapse=' '))
  }
  res
}


compare_docs_fotka = function(docs,n){
  my_tokenize = function(x) NGramTokenizer(x, Weka_control(min = n, max = n))
  
  docs<-VCorpus(VectorSource(docs))
  tdm <- TermDocumentMatrix(docs, control = list(tokenize = my_tokenize))
  cosine_dist_mat <- 1 - crossprod_simple_triplet_matrix(tdm)/(sqrt(col_sums(tdm^2) %*% t(col_sums(tdm^2))))
  cosine_dist_mat
}

docs.clean = prep_fun(docs)
docs.lemmas = lemmatize(docs.clean)
m=compare_docs_fotka(docs.lemmas,1)
diag(m)=NA