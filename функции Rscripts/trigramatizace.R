args = commandArgs(trailingOnly=TRUE)
library(textreuse)
library(rjson)
process_trigrams = function(doc,output){
  # creates path/id.json file in
  
  s=NULL
  tryCatch({
    s=tokenize_ngrams(doc,3)
  }, error=function(e){})
  if(!is.null(s)){
    f <- file(description=output,open="w",encoding="UTF-8") 
    cat(toJSON(list(trigrams = s)), file = f) 
    close(f) 
  }
}

f = args[1]
out = args[2]
doc = readChar(f, file.info(f)$size)

process_trigrams(doc,out)