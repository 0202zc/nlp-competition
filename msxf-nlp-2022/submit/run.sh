cd /usr/local/msxf2022-nlp && \
python3 data_prompt.py && \
python3 train.py --train_batch_size 12 --eval_batch_size 16 --do_train --do_eval && \
python3 predict.py --test_batch_size 50 && \
python3 export_results.py