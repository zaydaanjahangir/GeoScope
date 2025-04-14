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
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
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
        # Initialize model
        model = StreetCLIP(clip_model_version=args.clip_model).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create trainer
        trainer = StreetCLIPTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            output_dir=args.output_dir
        )
        
        # Create dataloaders using train-test split if only train_csv is provided
        if args.train_csv and not args.val_csv:
            train_dataloader, val_dataloader = create_train_test_split(
                args.train_csv,
                args.image_dir,
                test_size=0.2,  # 20% for validation
                batch_size=args.batch_size
            )
        else:
            # Use existing separate train/val files if provided
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
        
        # Train model
        trainer.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            num_epochs=args.num_epochs
        )
    
    elif args.mode == 'evaluate':
        # Load model
        model = StreetCLIP(clip_model_version=args.clip_model).to(device)
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create evaluator
        evaluator = StreetCLIPEvaluator(
            model=model,
            device=device,
            output_dir=args.output_dir
        )
        
        # Load candidates
        candidates = load_candidates(args.candidates_file)
        
        # Create test dataloader
        test_dataloader = create_dataloader(
            args.test_csv,
            args.image_dir,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        metrics = evaluator.evaluate(
            test_dataloader=test_dataloader,
            candidate_locations=candidates['locations']
        )
        
        # Save metrics
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main() 