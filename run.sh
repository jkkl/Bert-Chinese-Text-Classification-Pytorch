export PYTHONIOENCODING='utf-8'
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="-1" python -u run.py --model bert --task_name General_Intention_170_V1_binary --task_desc 170_binary --data_dir data/Intention170_binary
