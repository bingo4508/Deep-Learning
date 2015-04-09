Deep-Learning
========================

##Environment:
Linux with BLAS, LAPACK installed



##Package dependency:
1. Armadillo C++ linear algebra library
2. BLAS
3. LAPACK



##How to compile:
cd src
sh compile.sh



##How to run:
Before training, use the following to merge data and label, if label is not
0~N, please supply out_map to save the mapping of  string label to numeric
label(0~N).

`script/merge_data_label.py train.ark train.lab new_train.ark [out_map]`

Each epoch will save the model to output_model, if load_model supplied, will
load that model first and start training.

`bin/train learning_rate(0.01) batch_size(10) structure(5-4-3) max_epoch(100) new_train.ark output_model [load_model]`

if the .ark has answer(new_train.ark) has_answer=1, if not(test.ark)
has_answer=0
`bin/predict test_file model_name raw_result has_answer(1/0)`


##For Kaggle Sumission
48_phone->39_phone  

`map_phone_label.py raw_result final_result out_map 48-39.map`

1942_state->39_phone

`map_state_label.py raw_result final_result state-48-39.map`

Upload final_result
