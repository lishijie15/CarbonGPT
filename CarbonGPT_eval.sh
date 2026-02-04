# to fill in the following path to evaluation!
output_model=./checkpoints/CaronGPT_Encoder_7b_pv10
datapath=./ST_data/test_10pv/test_10pv_withUTN.json
st_data_path=./ST_data/test_10pv/test_pv10_withUTN.pkl
res_path=./result_test/CaronGPT_Encoder_7b_pv10_
start_id=0
end_id=593208
num_gpus=8

python ./CarbonGPT/eval/test_CarbonGPT.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}