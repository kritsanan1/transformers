"""
Web Application for Isan Pin AI

This module implements:
- FastAPI backend for Isan Pin music generation
- Gradio interface for user-friendly interaction
- REST API endpoints for programmatic access
- Real-time music generation and style transfer
- Audio upload and processing
- Results download and sharing
"""

import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import asyncio
import uvicorn

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from pydantic import BaseModel

from ..config import MODEL_CONFIG, AUDIO_CONFIG, STORAGE_CONFIG
from ..inference.inference import IsanPinInference
from ..evaluation.evaluator import IsanPinEvaluator
from ..utils.audio import AudioUtils

logger = logging.getLogger(__name__)

# Pydantic models for API
class GenerationRequest(BaseModel):
    description: str
    style: str = "lam_plearn"
    mood: str = None
    duration: float = 30.0
    num_samples: int = 1
    temperature: float = 1.0
    guidance_scale: float = 3.0

class StyleTransferRequest(BaseModel):
    target_style: str
    description: str = None
    strength: float = 0.7

class EvaluationRequest(BaseModel):
    reference_style: str = None
    reference_mood: str = None

class IsanPinWebApp:
    """Web application for Isan Pin AI"""
    
    def __init__(
        self,
        classifier_path: str = None,
        generator_path: str = None,
        cache_dir: str = None,
        temp_dir: str = None,
    ):
        """
        Initialize web application
        
        Args:
            classifier_path: Path to trained classifier
            generator_path: Path to fine-tuned generator
            cache_dir: Cache directory
            temp_dir: Temporary directory for files
        """
        self.cache_dir = cache_dir
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize inference system
        self.inference = IsanPinInference(
            classifier_path=classifier_path,
            generator_path=generator_path,
            cache_dir=cache_dir,
        )
        
        # Initialize evaluator
        self.evaluator = IsanPinEvaluator(
            classifier_path=classifier_path,
        )
        
        # Audio utilities
        self.audio_utils = AudioUtils()
        
        # Available styles and moods
        self.available_styles = list(MODEL_CONFIG["style_definitions"].keys())
        self.available_moods = [
            "sad", "happy", "contemplative", "romantic", "energetic", 
            "calm", "nostalgic", "joyful", "melancholic", "uplifting"
        ]
        
        logger.info("Isan Pin web application initialized")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="Isan Pin AI API",
    description="AI-powered generation of traditional Isan Pin music from Northeast Thailand",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global web app instance
web_app = None


@app.on_event("startup")
async def startup_event():
    """Initialize the web application"""
    global web_app
    
    logger.info("Starting Isan Pin AI web application...")
    
    # Initialize web app with default configuration
    web_app = IsanPinWebApp()
    
    logger.info("Isan Pin AI web application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    global web_app
    
    if web_app:
        web_app.cleanup_temp_files()
        logger.info("Isan Pin AI web application shutdown complete")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Isan Pin AI API",
        "version": "1.0.0",
        "description": "AI-powered generation of traditional Isan Pin music from Northeast Thailand",
        "docs": "/docs",
        "gradi": "/gradio",
    }


@app.get("/styles")
async def get_styles():
    """Get available music styles"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    styles = []
    for style_key, style_info in MODEL_CONFIG["style_definitions"].items():
        styles.append({
            "key": style_key,
            "thai_name": style_info.get("thai_name", style_key),
            "description": style_info.get("description", ""),
            "tempo_range": style_info.get("tempo_range", []),
            "moods": style_info.get("mood", []),
        })
    
    return {"styles": styles}


@app.get("/moods")
async def get_moods():
    """Get available moods"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {"moods": web_app.available_moods}


@app.post("/generate")
async def generate_music(request: GenerationRequest):
    """Generate music from text description"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Generate music
        results = web_app.inference.generate_music(
            description=request.description,
            style=request.style,
            mood=request.mood,
            duration=request.duration,
            num_samples=request.num_samples,
            temperature=request.temperature,
            guidance_scale=request.guidance_scale,
            post_process=True,
            quality_filter=True,
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Music generation failed")
        
        # Save results temporarily
        result_files = []
        for i, result in enumerate(results):
            # Save audio file
            audio_filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.wav"
            audio_path = web_app.temp_dir / audio_filename
            
            audio_to_save = result.get('audio_processed', result.get('audio'))
            web_app.inference.export_audio(
                audio_to_save,
                str(audio_path),
                format="wav",
                metadata={
                    "description": request.description,
                    "style": request.style,
                    "mood": request.mood,
                    "generated_at": datetime.now().isoformat(),
                }
            )
            
            result_files.append({
                "file_path": str(audio_path),
                "filename": audio_filename,
                "quality_score": result.get('quality_score'),
                "classification": result.get('classification'),
                "duration": result.get('duration'),
            })
        
        return {
            "message": "Music generated successfully",
            "results": result_files,
            "parameters": request.dict(),
        }
        
    except Exception as e:
        logger.error(f"Music generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Music generation failed: {str(e)}")


@app.post("/style-transfer")
async def style_transfer(
    audio_file: UploadFile = File(...),
    target_style: str = Form(...),
    description: str = Form(None),
    strength: float = Form(0.7),
):
    """Apply style transfer to uploaded audio"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Save uploaded file
        temp_input_path = web_app.temp_dir / f"uploaded_{audio_file.filename}"
        with open(temp_input_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Apply style transfer
        result = web_app.inference.style_transfer(
            audio=str(temp_input_path),
            target_style=target_style,
            description=description,
            strength=strength,
            post_process=True,
        )
        
        # Save result
        output_filename = f"style_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = web_app.temp_dir / output_filename
        
        web_app.inference.export_audio(
            result['audio'],
            str(output_path),
            format="wav",
            metadata={
                "original_file": audio_file.filename,
                "target_style": target_style,
                "strength": strength,
                "processed_at": datetime.now().isoformat(),
            }
        )
        
        return {
            "message": "Style transfer completed",
            "file_path": str(output_path),
            "filename": output_filename,
            "quality_score": result.get('quality_score'),
            "classification": result.get('classification'),
        }
        
    except Exception as e:
        logger.error(f"Style transfer failed: {e}")
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")


@app.post("/evaluate")
async def evaluate_audio(
    audio_file: UploadFile = File(...),
    reference_style: str = Form(None),
    reference_mood: str = Form(None),
):
    """Evaluate uploaded audio"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Save uploaded file
        temp_path = web_app.temp_dir / f"evaluate_{audio_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Evaluate
        results = web_app.evaluator.evaluate_single_audio(
            audio=str(temp_path),
            reference_style=reference_style,
            reference_mood=reference_mood,
            detailed=True,
        )
        
        return {
            "message": "Evaluation completed",
            "results": results,
            "filename": audio_file.filename,
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio file"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    file_path = web_app.temp_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav",
    )


# Gradio Interface

def create_gradio_interface():
    """Create Gradio interface"""
    
    def generate_music_gradio(
        description,
        style,
        mood,
        duration,
        num_samples,
        temperature,
        guidance_scale,
    ):
        """Generate music from Gradio interface"""
        if not web_app:
            return ["Service not initialized"] + [None] * 4
        
        try:
            results = web_app.inference.generate_music(
                description=description,
                style=style,
                mood=mood,
                duration=duration,
                num_samples=num_samples,
                temperature=temperature,
                guidance_scale=guidance_scale,
                post_process=True,
                quality_filter=True,
            )
            
            if not results:
                return ["No results generated"] + [None] * 4
            
            # Return first result
            result = results[0]
            audio = result.get('audio_processed', result.get('audio'))
            
            # Create temporary file
            temp_file = web_app.temp_dir / f"gradio_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            web_app.inference.export_audio(audio, str(temp_file), format="wav")
            
            quality_score = result.get('quality_score', 0)
            classification = result.get('classification', {})
            predicted_style = classification.get('predicted_style', 'unknown')
            confidence = classification.get('confidence', 0)
            
            return [
                f"Generated successfully!",
                str(temp_file),
                f"Quality Score: {quality_score:.3f}",
                f"Predicted Style: {predicted_style}",
                f"Confidence: {confidence:.3f}",
            ]
            
        except Exception as e:
            return [f"Error: {str(e)}"] + [None] * 4
    
    def style_transfer_gradio(
        audio_file,
        target_style,
        description,
        strength,
    ):
        """Apply style transfer from Gradio interface"""
        if not web_app or not audio_file:
            return ["Service not initialized or no audio file"] + [None] * 3
        
        try:
            # Apply style transfer
            result = web_app.inference.style_transfer(
                audio=audio_file,
                target_style=target_style,
                description=description,
                strength=strength,
                post_process=True,
            )
            
            audio = result.get('audio_processed', result.get('audio'))
            
            # Create temporary file
            temp_file = web_app.temp_dir / f"gradio_style_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            web_app.inference.export_audio(audio, str(temp_file), format="wav")
            
            quality_score = result.get('quality_score', 0)
            classification = result.get('classification', {})
            predicted_style = classification.get('predicted_style', 'unknown')
            
            return [
                f"Style transfer completed!",
                str(temp_file),
                f"Quality Score: {quality_score:.3f}",
                f"Predicted Style: {predicted_style}",
            ]
            
        except Exception as e:
            return [f"Error: {str(e)}"] + [None] * 3
    
    def evaluate_gradio(audio_file, reference_style, reference_mood):
        """Evaluate audio from Gradio interface"""
        if not web_app or not audio_file:
            return ["Service not initialized or no audio file"] + [None] * 3
        
        try:
            results = web_app.evaluator.evaluate_single_audio(
                audio=audio_file,
                reference_style=reference_style,
                reference_mood=reference_mood,
                detailed=False,
            )
            
            overall_score = results.get('overall_score', 0)
            basic_metrics = results.get('basic_metrics', {})
            quality_score = basic_metrics.get('quality_score', 0)
            
            style_consistency = results.get('style_consistency', {})
            consistency_score = style_consistency.get('overall_consistency', 0)
            
            return [
                f"Evaluation completed!",
                f"Overall Score: {overall_score:.3f}",
                f"Quality Score: {quality_score:.3f}",
                f"Style Consistency: {consistency_score:.3f}",
            ]
            
        except Exception as e:
            return [f"Error: {str(e)}"] + [None] * 3
    
    # Create Gradio interface
    with gr.Blocks(title="Isan Pin AI") as interface:
        gr.Markdown("# 🎵 Isan Pin AI - Traditional Thai Music Generation")
        gr.Markdown("Generate authentic traditional Isan Pin music from Northeast Thailand using AI.")
        
        with gr.Tab("Generate Music"):
            with gr.Row():
                with gr.Column():
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="Describe the Isan Pin music you want to generate...",
                        lines=3,
                    )
                    style_input = gr.Dropdown(
                        choices=web_app.available_styles if web_app else [],
                        value="lam_plearn",
                        label="Style",
                    )
                    mood_input = gr.Dropdown(
                        choices=web_app.available_moods if web_app else [],
                        value=None,
                        label="Mood (optional)",
                    )
                    duration_input = gr.Slider(
                        minimum=5,
                        maximum=120,
                        value=30,
                        step=5,
                        label="Duration (seconds)",
                    )
                    num_samples_input = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Number of Samples",
                    )
                    temperature_input = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature (creativity)",
                    )
                    guidance_scale_input = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    
                    generate_btn = gr.Button("Generate Music", variant="primary")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Status", interactive=False)
                    audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    quality_output = gr.Textbox(label="Quality Score", interactive=False)
                    style_output = gr.Textbox(label="Predicted Style", interactive=False)
                    confidence_output = gr.Textbox(label="Confidence", interactive=False)
        
        with gr.Tab("Style Transfer"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="Upload Audio", type="filepath")
                    target_style_input = gr.Dropdown(
                        choices=web_app.available_styles if web_app else [],
                        value="lam_plearn",
                        label="Target Style",
                    )
                    description_st_input = gr.Textbox(
                        label="Description (optional)",
                        placeholder="Describe the desired outcome...",
                    )
                    strength_input = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Transfer Strength",
                    )
                    
                    transfer_btn = gr.Button("Apply Style Transfer", variant="primary")
                
                with gr.Column():
                    status_st_output = gr.Textbox(label="Status", interactive=False)
                    audio_st_output = gr.Audio(label="Style-Transferred Audio", type="filepath")
                    quality_st_output = gr.Textbox(label="Quality Score", interactive=False)
                    style_st_output = gr.Textbox(label="Predicted Style", interactive=False)
        
        with gr.Tab("Evaluate Audio"):
            with gr.Row():
                with gr.Column():
                    audio_eval_input = gr.Audio(label="Upload Audio", type="filepath")
                    ref_style_eval_input = gr.Dropdown(
                        choices=web_app.available_styles if web_app else [],
                        value=None,
                        label="Expected Style (optional)",
                    )
                    ref_mood_eval_input = gr.Dropdown(
                        choices=web_app.available_moods if web_app else [],
                        value=None,
                        label="Expected Mood (optional)",
                    )
                    
                    evaluate_btn = gr.Button("Evaluate", variant="primary")
                
                with gr.Column():
                    status_eval_output = gr.Textbox(label="Status", interactive=False)
                    overall_eval_output = gr.Textbox(label="Overall Score", interactive=False)
                    quality_eval_output = gr.Textbox(label="Quality Score", interactive=False)
                    consistency_eval_output = gr.Textbox(label="Style Consistency", interactive=False)
        
        # Connect events
        generate_btn.click(
            fn=generate_music_gradio,
            inputs=[
                description_input,
                style_input,
                mood_input,
                duration_input,
                num_samples_input,
                temperature_input,
                guidance_scale_input,
            ],
            outputs=[
                status_output,
                audio_output,
                quality_output,
                style_output,
                confidence_output,
            ],
        )
        
        transfer_btn.click(
            fn=style_transfer_gradio,
            inputs=[
                audio_input,
                target_style_input,
                description_st_input,
                strength_input,
            ],
            outputs=[
                status_st_output,
                audio_st_output,
                quality_st_output,
                style_st_output,
            ],
        )
        
        evaluate_btn.click(
            fn=evaluate_gradio,
            inputs=[
                audio_eval_input,
                ref_style_eval_input,
                ref_mood_eval_input,
            ],
            outputs=[
                status_eval_output,
                overall_eval_output,
                quality_eval_output,
                consistency_eval_output,
            ],
        )
    
    return interface


# Mount Gradio interface to FastAPI
@app.get("/gradio")
async def gradio_interface():
    """Serve Gradio interface"""
    if not web_app:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    interface = create_gradio_interface()
    return interface.launch(prevent_thread_lock=True, quiet=True)


def run_web_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    classifier_path: str = None,
    generator_path: str = None,
    cache_dir: str = None,
):
    """
    Run the web application
    
    Args:
        host: Host to bind to
        port: Port to listen on
        classifier_path: Path to trained classifier
        generator_path: Path to fine-tuned generator
        cache_dir: Cache directory
    """
    logger.info(f"Starting Isan Pin AI web application on {host}:{port}")
    
    # Initialize global web app
    global web_app
    web_app = IsanPinWebApp(
        classifier_path=classifier_path,
        generator_path=generator_path,
        cache_dir=cache_dir,
    )
    
    # Run FastAPI
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Run the web application
    run_web_app()