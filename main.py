#!/usr/bin/env python3
"""
Isan Pin AI - Main Entry Point

This script provides a command-line interface for the Isan Pin AI system.
You can use it to:
- Collect and preprocess data
- Train models
- Generate new Pin music
- Evaluate results
- Launch web interface
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    get_version,
    get_config_summary,
    DataCollector,
    AudioClassifier,
    IsanPinMusicGen,
    IsanPinClassifier,
    IsanPinInference,
    IsanPinEvaluator,
    run_web_app,
)

def main():
    parser = argparse.ArgumentParser(
        description="Isan Pin AI - Generate traditional Isan Pin music using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data
  python main.py collect-data --input-dir ./raw_audio --output-dir ./processed
  
  # Train model
  python main.py train --data-dir ./dataset --output-dir ./models
  
  # Generate music
  python main.py generate --prompt "สร้างเสียงพิณแบบหมอลำเพลินช้าๆ เศร้าๆ" --duration 30
  
  # Launch web interface
  python main.py web --port 8000
  
  # Evaluate model
  python main.py evaluate --model-path ./models/best_model --test-data ./test_set
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Isan Pin AI {get_version()}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data collection command
    data_parser = subparsers.add_parser("collect-data", help="Collect and preprocess audio data")
    data_parser.add_argument("--input-dir", required=True, help="Input directory with raw audio files")
    data_parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    data_parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate for audio processing")
    data_parser.add_argument("--classify", action="store_true", help="Classify audio files automatically")
    
    # Training command
    train_parser = subparsers.add_parser("train-classifier", help="Train the Isan Pin audio classification model")
    train_parser.add_argument("--data-dir", required=True, help="Directory with training data")
    train_parser.add_argument("--output-dir", required=True, help="Output directory for trained models")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    
    # Generator training command
    gen_train_parser = subparsers.add_parser("train-generator", help="Train the Isan Pin music generation model")
    gen_train_parser.add_argument("--data-dir", required=True, help="Directory with training data")
    gen_train_parser.add_argument("--output-dir", required=True, help="Output directory for trained models")
    gen_train_parser.add_argument("--base-model", default="facebook/musicgen-small", help="Base model for fine-tuning")
    gen_train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    gen_train_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    gen_train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    
    # Generation command
    gen_parser = subparsers.add_parser("generate", help="Generate new Pin music")
    gen_parser.add_argument("--prompt", required=True, help="Text prompt describing the desired music")
    gen_parser.add_argument("--output-file", help="Output file path for generated audio")
    gen_parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    gen_parser.add_argument("--model-path", help="Path to trained model (uses default if not specified)")
    gen_parser.add_argument("--style", help="Specific style (lam_plearn, lam_sing, etc.)")
    gen_parser.add_argument("--tempo", help="Tempo (slow, medium, fast)")
    gen_parser.add_argument("--mood", help="Mood (sad, fun, romantic, etc.)")
    gen_parser.add_argument("--key", help="Musical key (C, D, E, etc.)")
    
    # Style transfer command
    transfer_parser = subparsers.add_parser("transfer-style", help="Transfer style between audio files")
    transfer_parser.add_argument("--input-audio", required=True, help="Input audio file")
    transfer_parser.add_argument("--target-style", required=True, help="Target style to transfer to")
    transfer_parser.add_argument("--output-file", required=True, help="Output file path")
    transfer_parser.add_argument("--content-weight", type=float, default=0.7, help="Content preservation weight")
    transfer_parser.add_argument("--style-weight", type=float, default=0.3, help="Style application weight")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model-path", required=True, help="Path to trained model")
    eval_parser.add_argument("--test-data", required=True, help="Test dataset path")
    eval_parser.add_argument("--output-dir", help="Output directory for evaluation results")
    eval_parser.add_argument("--human-evaluation", action="store_true", help="Include human evaluation")
    eval_parser.add_argument("--objective-metrics", action="store_true", help="Calculate objective metrics")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Launch web interface")
    web_parser.add_argument("--port", type=int, default=8000, help="Port to run the web server")
    web_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server")
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    web_parser.add_argument("--gradio", action="store_true", help="Use Gradio interface instead of FastAPI")
    
    # Configuration command
    config_parser = subparsers.add_parser("config", help="Show configuration information")
    config_parser.add_argument("--detailed", action="store_true", help="Show detailed configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == "config":
        print("Isan Pin AI Configuration")
        print("=" * 50)
        config_summary = get_config_summary()
        for key, value in config_summary.items():
            print(f"{key:20}: {value}")
        
        if args.detailed:
            print("\\nDetailed configuration available in src/config.py")
    
    elif args.command == "collect-data":
        print("Starting data collection...")
        collector = DataCollector(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
        )
        
        if args.classify:
            classifier = AudioClassifier()
            collector.set_classifier(classifier)
        
        collector.process_all()
        print(f"Data collection completed. Output saved to: {args.output_dir}")
    
    elif args.command == "train-classifier":
        print("Training classifier...")
        classifier = IsanPinClassifier()
        
        # This would load training data and train the classifier
        # For now, we'll just initialize it
        print("Classifier training would be implemented here")
        print("Use the classifier training functionality from the classification module")
    
    elif args.command == "train-generator":
        print("Training music generator...")
        generator = IsanPinMusicGen()
        
        # This would load training data and fine-tune the generator
        # For now, we'll just initialize it
        print("Generator training would be implemented here")
        print("Use the fine-tuning functionality from the musicgen module")
    
    elif args.command == "generate":
        print("Generating Pin music...")
        
        # Initialize inference system
        inference = IsanPinInference()
        
        # Generate music
        results = inference.generate_music(
            description=args.prompt,
            style=args.style,
            mood=args.mood,
            duration=args.duration,
            num_samples=1,
            temperature=1.0,
            guidance_scale=3.0,
        )
        
        if results:
            result = results[0]
            audio = result.get('audio_processed', result.get('audio'))
            
            output_file = args.output_file or "generated_pin.wav"
            inference.export_audio(audio, output_file)
            print(f"Generated music saved to: {output_file}")
        else:
            print("Music generation failed")
    
    elif args.command == "transfer-style":
        print("Transferring style...")
        
        # Initialize inference system
        inference = IsanPinInference()
        
        # Apply style transfer
        result = inference.style_transfer(
            audio=args.input_audio,
            target_style=args.target_style,
            description=None,
            strength=args.content_weight,
            post_process=True,
        )
        
        audio = result.get('audio_processed', result.get('audio'))
        inference.export_audio(audio, args.output_file)
        print(f"Style transfer completed. Output saved to: {args.output_file}")
    
    elif args.command == "evaluate":
        print("Evaluating model...")
        
        # Initialize evaluator
        evaluator = IsanPinEvaluator()
        
        # This would load test data and evaluate the model
        # For now, we'll just demonstrate the evaluation functionality
        print("Evaluation functionality would be implemented here")
        print("Use the evaluation functionality from the evaluator module")
        
        # Example: evaluate a single audio file
        if args.test_data and os.path.isfile(args.test_data):
            results = evaluator.evaluate_single_audio(
                audio=args.test_data,
                reference_style=None,
                reference_mood=None,
                detailed=True,
            )
            print(f"Evaluation results: {results}")
    
    elif args.command == "web":
        print(f"Starting web interface on {args.host}:{args.port}")
        
        run_web_app(
            host=args.host,
            port=args.port,
        )

if __name__ == "__main__":
    main()