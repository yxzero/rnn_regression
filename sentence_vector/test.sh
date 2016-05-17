python paragraph_data.py --getdata
./word2vec -train train.txt -output vectors.txt -cbow 0 -size 15 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
grep '_\*' vectors.txt > sentence_vectors.txt
