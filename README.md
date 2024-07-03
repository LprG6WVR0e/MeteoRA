# MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models

This is the implementation of the paper "MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models".

## Directory structure

- `base_model`: MeteoRA model
- `ckpt`: the datasets and dataset processing code.
- `eval`: the evaluation results and evaluation code.
- `MoELoRA`: MeteoRA module and adapted PEFT code.

## Usage

### Preparation

1. Install necessary packages:
```
pip install -r requirements.txt
```
2. Download all JSON format BIG-bench benchmarks, please refer to [here](https://huggingface.co/datasets/google/bigbench).
3. Change `bigbench_dataset_dir` path in `configs/config.yaml`.
4. Prepare datasets:
```
cd data
python create_dataset.py --task all
```

If you just want to create a specific dataset, run:
```
cd data
python create_dataset.py --task <task_name>
```
5. Prepare *composite-n* tasks:
```
python create_composite.py --n <n>
```
We prepared `n=3`, `n=5` and `n=10` few-shot dataset generating code. Before generating, please ensure that the sub-tasks to composite *composite-n* task have been included in `data/datasets`.

6. Prepare LoRA adapters checkpoint and MeteoRA model checkpoint. You can train by yourself or download ours([LlaMA2](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama2_13b) and [LlaMA3](https://huggingface.co/hDPQ4gi9BG/MeteoRA_llama3_8b) as base model) by:
```
python download_ckpt.py
```
7. Change other paths in `configs/config.yaml`. Example:
```yaml
base_model_path: 'meta-llama3/Meta-Llama-3-8B'
meteora_ckpt_path: 'ckpt/llama3_8b/llama3_8b_meteora/top_2'
adapter_dir: 'ckpt/llama3_8b/llama3_8b_peft'
```

### Evaluation

Running a benchmark with MeteoRA model:
```
python eval_model.py --task <task_name> --batch_size <batch_size> 
```

For example:
```
python eval_model.py --task composite_10 --batch_size 4 
```

**Note:** If you want to run a *composite-n* task, please set a larger *temperature* value (`self.T` in `MoELoRA/layer.py`). As a reference, `15`, `20` and `30` for `n=3`, `n=5` and `n=10`.

Save the evaluation result:
```
python eval_model.py --task <task_name> --batch_size <batch_size> --save
```

Debug mode (model output and ground truth will be shown in the console):
```
python eval_model.py --task <task_name> --batch_size <batch_size> --debug
```

Running a benchmark with PEFT model:
```
python eval_model.py --task <task_name> --batch_size <batch_size> --model <adapter_name>
```

### Train MeteoRA model

1. Prepare LoRA adapters and corresponding datasets in JSONL format. Ensure each LoRA adapter has a corresponding dataset. Place all LoRA adapters and datasets in their respective folders with matching subfolder names:
      ```
      - lora_adapters
            - adapter_name1
            - adapter_name2
            - ...
      - datasets
            - dataset_name1
            - dataset_name2
            - ...
      ```

2. Change file paths in `run_meteora_train_fsdp.sh`.

3. Train MeteoRA model:
```
sh run_meteora_train_fsdp.sh
```
