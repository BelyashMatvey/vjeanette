#!/bin/bash -l

#SBATCH --job-name=Task_train        # Job name
#SBATCH --output=Task_train.%j.log   # Standard output and error log 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=belyakov06mat@mail.ru
#SBATCH --cpus-per-task=32         # Run on a single CPU
#SBATCH --mem=128gb   
#SBATCH --constraint=gpu
cd ~/vidjil/seq2vdj
conda activate py_env
# /usr/bin/time -f "Реальное время: %e с\nПамять: %M КБ"  python -m vjeanette.run --in_file ~/for_igblast/m10245671.fastq --out_file ./out/test_out_10245671.csv --model_path ./weights/model_custom.pth  --logs ./logs/progress_10245671.log
/usr/bin/time -f "Реальное время: %e с\nПамять: %M КБ"  python -m vjeanette.run --in_file ~/for_igblast/m18748717.fastq --out_file ./out/test_out_18748717_2.csv --model_path ./weights/model_custom_2.pth  --logs ./logs/progress_18748717_2.log
# /usr/bin/time -f "Реальное время: %e с\nПамять: %M КБ"  python -m vjeanette.run --in_file ~/for_igblast/m18748718.fastq --out_file ./out/test_out_18748718.csv --model_path ./weights/model_custom.pth  --logs ./logs/progress_18748718.log
# /usr/bin/time -f "Реальное время: %e с\nПамять: %M КБ"  python -m vjeanette.run --in_file ~/for_igblast/m5494024.fastq --out_file ./out/test_out_5494024.csv --model_path ./weights/model_custom.pth  --logs ./logs/progress_5494024.log