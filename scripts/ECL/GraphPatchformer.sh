export CUDA_VISIBLE_DEVICES='1,2,3,4'

model_name=Graph-Patchformer

python -u run.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./datasets/ \
   --data custom \
   --data_path electricity.csv \
   --model_id electricity_96_96 \
   --model $model_name \
   --d_model 256 \
   --features M \
   --seq_len 96 \
   --pred_len 96 \
   --train_epochs 10 \
   --patience 3 \
   --e_layers 3 \
   --enc_in 321 \
   --dec_in 321 \
   --c_out 321 \
   --des 'Exp' \
   --d_ff 512 \
   --n_heads 16 \
   --learning_rate 0.001 \
   --dropout 0.2 \
   --batch_size 16 \
   --itr 1
