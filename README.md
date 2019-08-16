# Start
#### 準備訓練資料 `./pre-process-classify`
- 標注文字標籤的工具為[brat](https://brat.nlplab.org/)
1. 搜集檔案原始資料
    ###### 無法提供完整訓練集 盡請見諒

    ```buildoutcfg
    詢價	這個茶壺多少錢？
    其他	你們營業時間是幾點
    生氣	钱已经充到你们平台了，没金币玩屎呀
    詢價	買這整組多少錢
    詢價	買兩組的話多少
    生氣	你再開玩笑嗎？
    生氣	為什麼沒有打折？當初不是說只要800
    生氣	你們都是骗子嗎
    其他	妹仔多大啦
    ```

2. 輸出檔案

    ```buildoutcfg
    0	你好，有人在嗎？
    1	這個茶壺多少錢？
    2	你們營業時間是幾點
    3	钱已经充到你们平台了，没金币玩屎呀
    1	買這整組多少錢
    1	買兩組的話多少
    3	你再開玩笑嗎？
    3	為什麼沒有打折？當初不是說只要800
    3	你們都是骗子嗎
    2	妹仔多大啦
    0	店員在嗎？
    1	今日特價那組怎麼算
    1	300元能買多少？
    ```

#### 訓練模型 `./model_train`
1. 修改參數檔案

        ./model_train/run.sh:

        # 標籤數量
        --num_labels=4

        # 標籤編號
        --label_index=\{\"0\":\"問候\",\"1\":\"詢價\",\"2\":\"其他\",\"3\":\"生氣\"\}
2. 準備訓練資料

        # 把資料集分成訓練集與驗證集
        ./model_train/data/
        - for_train.csv
        - for_dev_and_test.csv
3. 執行

   重新訓練的話，記得要先清空舊檔案（output資料夾）
4. 輸出檔案

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


# Model

* ##### run.sh
    run the script to training, evaluation and prediction.
    
    ` sh run.sh `
    
    ```
    # path to pre-model folder
    # Download from https://github.com/ymcui/Chinese-BERT-wwm
    export BERT_BASE_DIR=/path/to/wwm/model/chinese-wwm_L-12_H-768_A-12
    
    # train data
    # data format:
    ##  <type int> <content>
    ##       0	  你好，有人在嗎？
    export GLUE_DIR=/path/to/train/data
    
    # output folder including model and result of evaluation and predication
    export TRAINED_CLASSIFIER=/path/to/output
    
    # main command
    ## num_labels : number of class of sentence
    ## label_index: chinese label of class, its for predication result
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

* ##### run_mobile.py

    definition of models
    
    * Processor object definition
    
        ```
        class customizedProcessor(DataProcessor):
          def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "for_train.csv")), "train")
        
          def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "for_dev_and_test.csv")), "dev")
        
          def get_test_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "for_dev_and_test.csv")), "test")
        
          def get_labels(self):
            """See base class."""
        
            labels = []
            for i in  range(FLAGS.num_labels):
              labels.append(str(i))
        
            return labels
        
          def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
              if i == 0:
                continue
              guid = "%s-%s" % (set_type, i)
              text_a = tokenization.convert_to_unicode(line[1])
              if set_type == "test":
                label = "0"
              else:
                label = tokenization.convert_to_unicode(line[0])
              examples.append(
                  InputExample(guid=guid, text_a=text_a, label=label))
            return examples
        
        ```
    
    * model
    
        ```
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)
        ```
    
    * do_train
        * estimator definition
    
            ```
            #
            if mode == tf.estimator.ModeKeys.TRAIN:
            
              train_op = optimization.create_optimizer(
                  total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            
              output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  train_op=train_op,
                  scaffold_fn=scaffold_fn)
            ```
    
        * get training data and initial variables.
    
            ```
            #
            if FLAGS.do_train:
                train_examples = processor.get_train_examples(FLAGS.data_dir)
                num_train_steps = int(
                    len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
                num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
                train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
                file_based_convert_examples_to_features(
                    train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
                train_input_fn = file_based_input_fn_builder(
                    input_file=train_file,
                    seq_length=FLAGS.max_seq_length,
                    is_training=True,
                    drop_remainder=True)
            ```
    
        * main process
    
            ```
            if FLAGS.do_train:
            
                tf.logging.info("***** Running training *****")
                tf.logging.info("  Num examples = %d", len(train_examples))
                tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
                tf.logging.info("  Num steps = %d", num_train_steps)
                estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
            
            ```
   
    * do_eval
    
        * estimator definition
    
            ```
            elif mode == tf.estimator.ModeKeys.EVAL:
            
                def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            
                    # add precision,recall,f1
                    precision = tf.metrics.precision(labels=label_ids, predictions=predictions, weights=is_real_example)
                    recall = tf.metrics.recall(labels=label_ids, predictions=predictions, weights=is_real_example)
                    f1 = (2 * precision[0] * recall[0] / (precision[0] + recall[0]),recall[1])
                    accuracy = tf.metrics.accuracy(
                        labels=label_ids, predictions=predictions, weights=is_real_example)
                    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_precision": precision,
                        "eval_recall": recall,
                        "eval_f1": f1,
                        "eval_loss": loss,
                    }
                ## modify end
            
              eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
              output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  eval_metrics=eval_metrics,
                  scaffold_fn=scaffold_fn)
            ```
    
        * get training data and initial variables.
    
            ```
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            num_actual_eval_examples = len(eval_examples)
            if FLAGS.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on. These do NOT count towards the metric (all tf.metrics
                # support a per-instance weight, and these get a weight of 0.0).
                while len(eval_examples) % FLAGS.eval_batch_size != 0:
                    eval_examples.append(PaddingInputExample())
            
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
            
            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(eval_examples), num_actual_eval_examples,
                            len(eval_examples) - num_actual_eval_examples)
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
            
            # This tells the estimator to run through the entire set.
            eval_steps = None
            
            # However, if running eval on the TPU, you will need to specify the
            # number of steps.
            if FLAGS.use_tpu:
              assert len(eval_examples) % FLAGS.eval_batch_size == 0
              eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
            
            eval_drop_remainder = True if FLAGS.use_tpu else False
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)
            ```
            
        * main process
    
            ```
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            
            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            ```
    
    * do_predication
    
        * estimator definition
    
            ```
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                      mode=mode,
                      predictions={"probabilities": probabilities},
                      scaffold_fn=scaffold_fn)
            
            ```
    
        * get training data and initial variables.
    
            ```
            predict_examples = processor.get_test_examples(FLAGS.data_dir)
            num_actual_predict_examples = len(predict_examples)
            if FLAGS.use_tpu:
              # TPU requires a fixed batch size for all batches, therefore the number
              # of examples must be a multiple of the batch size, or else examples
              # will get dropped. So we pad with fake examples which are ignored
              # later on.
              while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
            
            predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
            file_based_convert_examples_to_features(predict_examples, label_list,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file)
            
            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(predict_examples), num_actual_predict_examples,
                            len(predict_examples) - num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
            
            predict_drop_remainder = True if FLAGS.use_tpu else False
            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)
            
            source_array =[]
            with open(os.path.join(FLAGS.data_dir, "for_dev_and_test.csv")) as source_data:
            
              for i in source_data:
                source_array.append(i)
            
            source_array =source_array[1:]
            
            ```
    
        * main process
    
            ```
            result = estimator.predict(input_fn=predict_input_fn)
            output_predict_file = os.path.join(FLAGS.output_dir, "test_samples_results.csv")
            with open(output_predict_file,"w") as csvfile:
              num_written_lines = 0
              tf.logging.info("***** Predict results *****")
              for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                probabilities_label =  str(list(probabilities).index(max(probabilities)))
                if i >= num_actual_predict_examples:
                  break
                output_line = " ".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
            
                ans_full_content = source_array[i].strip("\n").replace("\t"," ")
                ans_label = ans_full_content[0]
                ans_text = ans_full_content[1:]
                label_chinese=json.loads(FLAGS.label_index)#
                if probabilities_label == ans_label:
                    output_line = ["o","答案: "+ans_label,label_chinese[ans_label],ans_text,"預測: "+probabilities_label,label_chinese[probabilities_label],output_line]
                else:
                    output_line = ["x","答案: "+ans_label,label_chinese[ans_label],ans_text,"預測: "+probabilities_label,label_chinese[probabilities_label],output_line]
            
                spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(output_line)
                num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples
            ```
