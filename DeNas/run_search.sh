RANDOM_SEED=`date +%s`
export OMP_NUM_THREADS=18
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"


# search bert model
# python -u search.py --domain bert --conf ../conf/denas/nlp/aidk_denas_bert.conf 2>&1 | tee ./log/search_bert_${RANDOM_SEED}.log

# search roberta model
python -u search.py --domain bert --conf ../conf/denas/nlp/aidk_denas_roberta.conf 2>&1 | tee ./log/search_roberta_${RANDOM_SEED}.log
