import argparse
import torch
import os
from typing import Dict, List
import json

from models.streetclip import StreetCLIP
from data.dataset import create_dataloader, create_train_test_split
from training.trainer import StreetCLIPTrainer
from evaluation.evaluator import StreetCLIPEvaluator
from utils.synthetic_caption import SyntheticCaptionGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='StreetCLIP Training and Evaluation')
    
    # General arguments
    parser.add_argument('--mode', choices=['train', 'evaluate', 'generate_captions'],
                       required=True, help='Mode of operation')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save outputs')
    
    # Data arguments
    parser.add_argument('--train_csv', help='Path to training CSV')
    parser.add_argument('--val_csv', help='Path to validation CSV')
    parser.add_argument('--test_csv', help='Path to test CSV')
    parser.add_argument('--image_dir', required=True,
                       help='Directory containing images')
    
    # Model arguments
    parser.add_argument('--clip_model', default='ViT-L/14',
                       help='CLIP model version')
    parser.add_argument('--checkpoint_path', help='Path to model checkpoint')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=12,
                       help='Accumulate gradients over this many steps')
    parser.add_argument('--warmup_epochs', type=float, default=0.6,
                       help='Fractional epochs to warm up learning rate')
    parser.add_argument('--geo_eval_frequency', type=int, default=1,
                       help='Evaluate geographic accuracy every N epochs')
    
    # Evaluation arguments
    parser.add_argument('--candidates_file', help='Path to candidates JSON file')
    
    return parser.parse_args()

def load_candidates(file_path: str) -> Dict:
    """Load candidate locations from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    if args.mode == 'generate_captions':
        # Generate captions for CSV
        SyntheticCaptionGenerator.process_csv(
            args.train_csv,
            os.path.join(args.output_dir, 'train_with_captions.csv')
        )
        if args.val_csv:
            SyntheticCaptionGenerator.process_csv(
                args.val_csv,
                os.path.join(args.output_dir, 'val_with_captions.csv')
            )
        if args.test_csv:
            SyntheticCaptionGenerator.process_csv(
                args.test_csv,
                os.path.join(args.output_dir, 'test_with_captions.csv')
            )

    elif args.mode == 'train':
        # Initialize model and move it to device *before* making optimizer
        model = StreetCLIP(clip_model_version=args.clip_model).to(device)

        # Create optimizer (now sees CUDA parameters)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Create trainer, passing all hyperparams
        trainer = StreetCLIPTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_epochs=args.warmup_epochs
        )

        # Build dataloaders
        if args.train_csv and not args.val_csv:
            train_dataloader, val_dataloader = create_train_test_split(
                args.train_csv,
                args.image_dir,
                test_size=0.2,
                batch_size=args.batch_size
            )
        else:
            train_dataloader = create_dataloader(
                args.train_csv,
                args.image_dir,
                batch_size=args.batch_size
            )
            val_dataloader = None
            if args.val_csv:
                val_dataloader = create_dataloader(
                    args.val_csv,
                    args.image_dir,
                    batch_size=args.batch_size
                )

        # Launch training
        trainer.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            num_epochs=args.num_epochs,
            geo_eval_frequency=args.geo_eval_frequency
        )

        del trainer
        del model
        del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    elif args.mode == 'evaluate':
        # Load model + checkpoint
        model = StreetCLIP(clip_model_version=args.clip_model).to(device)
        if args.checkpoint_path:
            chk = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(chk['model_state_dict'])

        # Create evaluator
        evaluator = StreetCLIPEvaluator(
            model=model,
            device=device,
            output_dir=args.output_dir
        )

        # Load candidates and test dataloader
        candidates = load_candidates(args.candidates_file)
        test_dataloader = create_dataloader(
            args.test_csv,
            args.image_dir,
            batch_size=args.batch_size
        )

        # Evaluate and save metrics
        metrics = evaluator.evaluate(
            test_dataloader=test_dataloader,
            candidate_locations=candidates['locations']
        )
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
