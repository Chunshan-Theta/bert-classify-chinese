export BERT_BASE_DIR=/Users/gavinwang/Desktop/BERT-train2deploy/base_model/chinese-wwm_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/Users/gavinwang/Desktop/BERT-train2deploy/output
python3 freeze_graph.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $TRAINED_CLASSIFIER \
    -max_seq_len 32 \
    -count_labels 2
