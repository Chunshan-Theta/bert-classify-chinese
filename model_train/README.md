
## 訓練前準備
1. 將預模型放在`base_model`

    > 使用模型的下載來源-> [下載](https://github.com/ymcui/Chinese-BERT-wwm)
    
    ```buildoutcfg
    ./base_model
      |- chinese-wwm_L-12_H-768_A-12
        |- bert_config.json
        |- bert_model.ckpt.data-00000-of-00001
        |- bert_model.ckpt.index
        |- bert_model.ckpt.meta
        |- vocab.txt
    ```
2. 將準備好的資料放在`data`資料夾中

    ```buildoutcfg
    ./data
        |- for_dev_and_test.csv
        |- for_train.csv
    ```
3. 調整腳本

    ` vim run.sh `
    
    要調整的部分有：
    * num_labels 標籤數量
    * label_index 標籤編號與對應中文標籤
    
    ```
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
      --label_index=\{\"0\":\"問候\",\"1\":\"詢價\",\"2\":\"其他\",\"3\":\"生氣\"\}

    ```
## 執行訓練
    
```sh run.sh```

* 部分過程輸出：
    ```buildoutcfg
    I0817 01:06:30.408478 140735690490752 estimator.py:1147] Done calling model_fn.
    I0817 01:06:30.410821 140735690490752 basic_session_run_hooks.py:541] Create CheckpointSaverHook.
    I0817 01:06:35.448922 140735690490752 monitored_session.py:240] Graph was finalized.
    I0817 01:06:42.216624 140735690490752 session_manager.py:500] Running local_init_op.
    I0817 01:06:42.493341 140735690490752 session_manager.py:502] Done running local_init_op.
    I0817 01:06:54.641889 140735690490752 basic_session_run_hooks.py:606] Saving checkpoints for 0 into ./output/model.ckpt.
    I0817 01:07:29.678798 140735690490752 tpu_estimator.py:2159] global_step/sec: 0.0689326
    I0817 01:07:29.679122 140735690490752 tpu_estimator.py:2160] examples/sec: 1.10292
    W0817 01:07:29.679260 140735690490752 basic_session_run_hooks.py:724] It seems that global step (tf.train.get_global_step) has not been increased. Current value (could be stable): 1 vs previous value: 1. You could increase the global step by passing tf.train.get_global_step() to Optimizer.apply_gradients or Optimizer.minimize.
    I0817 01:07:38.968907 140735690490752 tpu_estimator.py:2159] global_step/sec: 0.107641
    I0817 01:07:38.969228 140735690490752 tpu_estimator.py:2160] examples/sec: 1.72226
    I0817 01:07:48.229242 140735690490752 tpu_estimator.py:2159] global_step/sec: 0.107988
    I0817 01:07:48.229606 140735690490752 tpu_estimator.py:2160] examples/sec: 1.72781
    ```

* 結果觀察 
    * ./output/eval_results.txt

    ```buildoutcfg
    # 範例格式.
    eval_accuracy = 0.8792814
    eval_f1 = 0.8148148
    ```
  
    * ./output/test_samples_results.csv
    ```buildoutcfg
    # 範例格式
    o,答案: 1,詢價,這個茶壺多少錢？,預測: 1,詢價,"0.29443625 0.31483293 0.24977936 0.14095142
    ```