**Bunch of Rscripts that were used during FakeHacks**

to run them you need to:
1. install R
2. run `Rscript install.R` - to install relevant packages
3. run individual functions within scripts, no documentation provided yet

trigramatizace.R transforms a string (e.g. a news article) into a set of trigrams and outputs that as json file... run `Rscript trigramatizace.R "file.txt" "file.json" ` , the first argument is the input txt file, the second the output path
