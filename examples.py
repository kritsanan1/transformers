#!/usr/bin/env python3
"""
Isan Pin AI - Example Usage Script

This script demonstrates how to use the Isan Pin AI system for:
1. Generating traditional Isan Pin music
2. Applying style transfer
3. Evaluating generated music
4. Batch processing
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    IsanPinInference,
    IsanPinEvaluator,
    IsanPinClassifier,
    IsanPinMusicGen,
)

def example_generate_music():
    """Example: Generate traditional Isan Pin music"""
    print("🎵 Example 1: Generate Traditional Isan Pin Music")
    print("=" * 50)
    
    # Initialize inference system (using base models)
    inference = IsanPinInference()
    
    # Generate music with different styles
    styles = ["lam_plearn", "lam_sing", "lam_klorn"]
    descriptions = [
        "Traditional Isan Pin music for meditation and relaxation",
        "Fast and energetic Isan Pin music for dancing",
        "Poetic Isan Pin music with storytelling elements"
    ]
    
    for style, description in zip(styles, descriptions):
        print(f"\nGenerating {style} style...")
        
        try:
            results = inference.generate_music(
                description=description,
                style=style,
                duration=15.0,  # Shorter for demo
                num_samples=1,
                temperature=0.8,
                guidance_scale=3.0,
            )
            
            if results:
                result = results[0]
                audio = result.get('audio_processed', result.get('audio'))
                quality_score = result.get('quality_score', 0)
                classification = result.get('classification', {})
                
                print(f"✅ Generated {style} style music")
                print(f"   Quality Score: {quality_score:.3f}")
                print(f"   Predicted Style: {classification.get('predicted_style', 'unknown')}")
                print(f"   Confidence: {classification.get('confidence', 0):.3f}")
                
                # Save the generated audio
                output_file = f"example_generated_{style}.wav"
                inference.export_audio(audio, output_file)
                print(f"   Saved to: {output_file}")
            else:
                print(f"❌ Failed to generate {style} style music")
                
        except Exception as e:
            print(f"❌ Error generating {style}: {e}")
    
    print("\nExample 1 completed!\n")

def example_style_transfer():
    """Example: Apply style transfer to audio"""
    print("🔄 Example 2: Style Transfer Between Isan Pin Styles")
    print("=" * 50)
    
    # For this example, we'll create a simple test audio
    # In practice, you would use an existing audio file
    inference = IsanPinInference()
    
    # First, generate some music in one style
    print("Generating initial music in lam_plearn style...")
    results = inference.generate_music(
        description="Slow contemplative Isan Pin music",
        style="lam_plearn",
        duration=10.0,
        num_samples=1,
    )
    
    if not results:
        print("❌ Failed to generate initial music")
        return
    
    original_audio = results[0].get('audio_processed', results[0].get('audio'))
    
    # Apply style transfer to different styles
    target_styles = ["lam_sing", "lam_klorn"]
    
    for target_style in target_styles:
        print(f"\nTransferring to {target_style} style...")
        
        try:
            result = inference.style_transfer(
                audio=original_audio,
                target_style=target_style,
                description=f"Music in {target_style} style",
                strength=0.7,
                post_process=True,
            )
            
            transferred_audio = result.get('audio_processed', result.get('audio'))
            quality_score = result.get('quality_score', 0)
            classification = result.get('classification', {})
            
            print(f"✅ Transferred to {target_style} style")
            print(f"   Quality Score: {quality_score:.3f}")
            print(f"   Predicted Style: {classification.get('predicted_style', 'unknown')}")
            
            # Save the transferred audio
            output_file = f"example_transferred_{target_style}.wav"
            inference.export_audio(transferred_audio, output_file)
            print(f"   Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error transferring to {target_style}: {e}")
    
    print("\nExample 2 completed!\n")

def example_evaluation():
    """Example: Evaluate generated music"""
    print("📊 Example 3: Evaluate Music Quality and Authenticity")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = IsanPinEvaluator()
    
    # Generate some test music
    inference = IsanPinInference()
    print("Generating test music for evaluation...")
    
    results = inference.generate_music(
        description="Traditional Isan Pin music for evaluation",
        style="lam_plearn",
        duration=20.0,
        num_samples=1,
    )
    
    if not results:
        print("❌ Failed to generate test music")
        return
    
    test_audio = results[0].get('audio_processed', results[0].get('audio'))
    
    # Evaluate the music
    print("\nEvaluating generated music...")
    
    try:
        evaluation_results = evaluator.evaluate_single_audio(
            audio=test_audio,
            reference_style="lam_plearn",
            reference_mood="contemplative",
            detailed=True,
        )
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Overall Score: {evaluation_results.get('overall_score', 0):.3f}")
        
        # Basic metrics
        basic_metrics = evaluation_results.get('basic_metrics', {})
        print(f"   Quality Score: {basic_metrics.get('quality_score', 0):.3f}")
        print(f"   Duration: {basic_metrics.get('duration', 0):.1f}s")
        print(f"   RMS Energy: {basic_metrics.get('rms', 0):.3f}")
        
        # Style consistency
        style_consistency = evaluation_results.get('style_consistency', {})
        print(f"   Style Consistency: {style_consistency.get('overall_consistency', 0):.3f}")
        print(f"   Tempo Consistency: {style_consistency.get('tempo_consistency', 0):.3f}")
        
        # Cultural authenticity
        cultural_auth = evaluation_results.get('cultural_authenticity', {})
        print(f"   Cultural Authenticity: {cultural_auth.get('cultural_authenticity_score', 0):.3f}")
        print(f"   Tempo Authenticity: {cultural_auth.get('tempo_authenticity', 0):.3f}")
        print(f"   Rhythmic Authenticity: {cultural_auth.get('rhythmic_authenticity', 0):.3f}")
        
        # Audio quality
        audio_quality = evaluation_results.get('audio_quality', {})
        print(f"   Audio Quality: {audio_quality.get('audio_quality_score', 0):.3f}")
        print(f"   SNR (dB): {audio_quality.get('snr_db', 0):.1f}")
        
        # Save evaluation results
        import json
        with open('example_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"\n   Saved detailed results to: example_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    
    print("\nExample 3 completed!\n")

def example_batch_processing():
    """Example: Batch processing multiple descriptions"""
    print("📦 Example 4: Batch Processing Multiple Descriptions")
    print("=" * 50)
    
    inference = IsanPinInference()
    
    # Multiple descriptions for batch processing
    descriptions = [
        "Slow meditative Isan Pin music for relaxation",
        "Fast energetic Isan Pin music for celebration", 
        "Romantic Isan Pin music for evening atmosphere",
        "Contemplative Isan Pin music for deep thinking",
    ]
    
    styles = ["lam_plearn", "lam_sing", "lam_plearn", "lam_klorn"]
    
    print(f"Processing {len(descriptions)} descriptions in batch...")
    
    try:
        # Batch generate music
        all_results = inference.batch_generate(
            descriptions=descriptions,
            styles=styles,
            duration=10.0,  # Shorter for demo
            num_samples_per_description=1,
            max_workers=2,  # Limit parallel workers
            post_process=True,
        )
        
        print(f"\n✅ Batch processing completed!")
        print(f"   Generated {len(all_results)} sets of music")
        
        # Analyze results
        quality_scores = []
        predicted_styles = []
        
        for i, (desc, results) in enumerate(zip(descriptions, all_results)):
            if results:
                result = results[0]
                quality_score = result.get('quality_score', 0)
                classification = result.get('classification', {})
                predicted_style = classification.get('predicted_style', 'unknown')
                
                quality_scores.append(quality_score)
                predicted_styles.append(predicted_style)
                
                print(f"\n   Description {i+1}: {desc[:50]}...")
                print(f"   Expected Style: {styles[i]}")
                print(f"   Quality Score: {quality_score:.3f}")
                print(f"   Predicted Style: {predicted_style}")
                
                # Save audio
                output_file = f"example_batch_{i+1}_{styles[i]}.wav"
                audio = result.get('audio_processed', result.get('audio'))
                inference.export_audio(audio, output_file)
                print(f"   Saved to: {output_file}")
        
        # Summary statistics
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"\n📊 Batch Statistics:")
            print(f"   Average Quality: {avg_quality:.3f}")
            print(f"   Quality Range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
            
            # Style accuracy
            correct_styles = sum(1 for exp, pred in zip(styles, predicted_styles) if exp == pred)
            style_accuracy = correct_styles / len(styles) if styles else 0
            print(f"   Style Accuracy: {style_accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
    
    print("\nExample 4 completed!\n")

def main():
    """Main function to run all examples"""
    print("🎵 Isan Pin AI - Example Usage Script")
    print("=" * 60)
    print("This script demonstrates the capabilities of Isan Pin AI system")
    print("for generating and evaluating traditional Thai music.\n")
    
    # Check if models are available (this is a simplified check)
    print("🔍 Checking system status...")
    
    try:
        # Try to initialize the basic systems
        inference = IsanPinInference()
        evaluator = IsanPinEvaluator()
        print("✅ System initialized successfully")
        print("✅ Ready to generate and evaluate Isan Pin music\n")
    except Exception as e:
        print(f"⚠️  System initialization warning: {e}")
        print("⚠️  Some features may be limited without trained models\n")
    
    # Run examples
    try:
        example_generate_music()
        example_style_transfer()
        example_evaluation()
        example_batch_processing()
        
        print("🎉 All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("   1. Train models on real Isan Pin music data")
        print("   2. Fine-tune for better cultural authenticity")
        print("   3. Launch the web interface: python main.py web")
        print("   4. Explore the API documentation")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()