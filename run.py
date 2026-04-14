import argparse
import torch
from model.model import UNet1D_Embed
from inference.core import InferenceRunner
from inference.config import DEFAULT_THRESHOLDS

def main():
    parser = argparse.ArgumentParser('Fastq Inference Tool')
    parser.add_argument('--in_file',default = './data/test.fastq')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_cores', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32768)
    parser.add_argument('--logs', default='./logs/progress.log')
    parser.add_argument('--out_file', default='./out/test_out.csv')
    parser.add_argument('--model_path', default='./model/model_v1.pth')
    
    args = parser.parse_args()
    
    # Model adding
    model = UNet1D_Embed().to(args.device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model = torch.compile(model, mode="reduce-overhead")
    model = model.to(memory_format=torch.channels_last)
    
    # Inference start
    runner = InferenceRunner(model, args.device, DEFAULT_THRESHOLDS)
    runner.add_log_handler(args.logs)
    runner.run(
        fastq_file=args.in_file,
        output_csv=args.out_file,
        batch_size=args.batch_size,
        num_workers=args.n_cores
    )

if __name__ == "__main__":
    main()