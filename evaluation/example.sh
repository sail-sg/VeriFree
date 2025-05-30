
## evaluate Qwen3-8B-Base on GPQA-Diamond
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_gpqa.py --model_path Qwen3-8B-Base --n_repeats 10

## evaluate Qwen3-8B (w/o thinking) on GPQA-Diamond
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_gpqa.py --model_path Qwen3-8B --n_repeats 10 --disable_thinking

## evaluate Qwen3-8B (w/ thinking) on GPQA-Diamond
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_gpqa.py --model_path Qwen3-8B --n_repeats 10

## evaluate Qwen3-8B-Base-VeriFree on GPQA-Diamond
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_gpqa.py --model_path zhouxiangxin/Qwen3-8B-Base-VeriFree --n_repeats 10
