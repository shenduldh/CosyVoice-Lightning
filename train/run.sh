# python prepare_data.py \
#     --data_dirs ./test_samples \
#     --output_dir ./output \
#     --campplus ../assets/CosyVoice2-0.5B/campplus.onnx \
#     --speech_tokenizer ../assets/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
#     --audio_suffix .wav \
#     --text_suffix .txt \
#     --reject_audio_suffix .reject.wav \
#     --apply_dpo true

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc-per-node=2 --master-port=12345 train.py \
      --deepspeed_config ./ds_config.json \
      --deepspeed.save_states model+optimizer \
