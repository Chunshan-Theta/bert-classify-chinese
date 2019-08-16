export BERT_BASE_DIR=./base_model/chinese-wwm_L-12_H-768_A-12
export GLUE_DIR=./data
export TRAINED_CLASSIFIER=./output

python3 run_mobile.py \
  --task_name=customized \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=3 \
  --output_dir=$TRAINED_CLASSIFIER \
  --num_labels=4 \
  --label_index=\{\"0\":\"h_未到帳\",\"1\":\"h_充值\",\"2\":\"h_已提供\",\"3\":\"h_咒罵\"\}
