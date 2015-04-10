test_ark=MLDS_HW1_RELEASE_v1/fbank/test.ark
model=output/model
tmp_result=output/tmp_result
final_result=output/final_result
map1=MLDS_HW1_RELEASE_v1/fbank/train_merge.map
##################################################
map2=MLDS_HW1_RELEASE_v1/phones/48_39.map

bin/predict $test_ark $model $tmp_result
python script/map_result_label.py $tmp_result $final_result $map1 $map2
rm -f $tmp_result
