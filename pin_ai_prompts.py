#!/usr/bin/env python3
"""
Isan Pin AI Prompts Collection
Comprehensive prompts for Thai-Isan traditional music AI generation
"""

# Data Collection & Preparation Prompts
DATA_COLLECTION_PROMPTS = {
    "audio_classification": """คุณเป็น AI ผู้เชี่ยวชาญด้านดนตรีอีสานและการประมวลผลเสียง
งานของคุณคือช่วยจัดหมวดหมู่และติด metadata สำหรับไฟล์เสียงพิณอีสาน

สำหรับแต่ละไฟล์เสียง ให้วิเคราะห์และระบุ:
1. สไตล์การเล่น: [หมอลำเพลิน, หมอลำซิ่ง, หมอลำกลอน, ลำตัด, ลำพื้น]
2. Tempo: [ช้า (60-80 BPM), ปานกลาง (80-120 BPM), เร็ว (120-160 BPM)]
3. อารมณ์: [เศร้า, สนุกสนาน, โรแมนติก, ตื่นเต้น, เฉยๆ]
4. ภูมิภาค: [อีสานเหนือ, อีสานใต้, อีสานกลาง]
5. เทคนิคพิเศษ: [การสั่น, การเล่นเร็ว, การเด้ง, การประสาน]
6. คีย์หลัก: [C, D, E, G, A, etc.]
7. จังหวะ: [4/4, 3/4, 6/8, อิสระ]

Format output เป็น JSON:
{
  "filename": "pin_001.wav",
  "style": "lam_plearn",
  "tempo": 85,
  "mood": "romantic",
  "region": "northeast_north",
  "techniques": ["vibrato", "fast_plucking"],
  "key": "D",
  "time_signature": "4/4",
  "duration": 180,
  "quality_rating": 8.5
}""",

    "dataset_creation": """สร้าง training dataset สำหรับโมเดล AI เรียนรู้การเล่นพิณ:

ข้อกำหนด:
- จำนวนตัวอย่าง: 1000 ไฟล์
- ความยาวเฉลี่ย: 30-180 วินาที
- Sample rate: 48kHz
- Bit depth: 24-bit
- Format: WAV (uncompressed)

สำหรับแต่ละไฟล์ ต้องมี:
1. ไฟล์เสียงต้นฉบับ (original_audio.wav)
2. Text description (description.txt)
3. Metadata (metadata.json)
4. Spectrogram (spectrogram.png)
5. MIDI transcription (ถ้าเป็นไปได้)

Text description format:
"This is [style] style Isan Pin music, playing at [tempo] tempo with [mood] mood. 
The piece features [techniques] and is in [key] key. 
Notable characteristics include [special_features]."

ให้สร้าง Python script สำหรับ:
1. แปลงไฟล์เสียงเป็น format ที่ต้องการ
2. สร้าง spectrogram
3. แยกข้อมูล metadata
4. จัดโครงสร้างโฟลเดอร์"""
}

# Model Training Prompts
TRAINING_PROMPTS = {
    "musicgen_finetune": """Task: Fine-tune Facebook MusicGen model สำหรับการสร้างเสียงพิณอีสาน

Model Architecture:
- Base Model: facebook/musicgen-small
- Task: Conditional Audio Generation
- Input: Text description + optional audio prompt
- Output: Generated Pin audio (32kHz, mono)

Training Configuration:
{
  "model_name": "musicgen-pin-isan",
  "base_model": "facebook/musicgen-small",
  "learning_rate": 5e-5,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "num_epochs": 50,
  "warmup_steps": 1000,
  "max_length": 1500,
  "scheduler": "cosine_with_restarts",
  "optimizer": "adamw",
  "weight_decay": 0.01
}

Dataset Structure:
- Training samples: 800 files
- Validation samples: 150 files
- Test samples: 50 files

Text Prompt Templates สำหรับ Training:
1. "Generate {style} Isan Pin music with {tempo} tempo and {mood} mood"
2. "Create {region} style Pin performance featuring {technique}"
3. "Produce Pin music in {key} key with {time_signature} time signature"
4. "Synthesize traditional {style} Pin melody with modern arrangement"
5. "Compose emotional Pin piece inspired by {reference_song}"

Data Augmentation:
- Pitch shift: ±2 semitones
- Time stretch: 0.9x to 1.1x
- Add background noise: SNR 30-40dB
- Mix with other traditional instruments (optional)

Loss Functions:
- Primary: Multi-scale spectral loss
- Secondary: Perceptual loss (using pre-trained audio classifier)
- Tertiary: Style consistency loss

Evaluation Metrics:
1. Objective:
   - Frechet Audio Distance (FAD)
   - Multi-scale STFT loss
   - Signal-to-Noise Ratio (SNR)
   - Mel Cepstral Distortion (MCD)

2. Subjective:
   - Authenticity score (1-10)
   - Musical quality (1-10)
   - Style accuracy (1-10)
   - Overall satisfaction (1-10)
""",

    "training_script": """สร้าง complete training script สำหรับโมเดลเรียนรู้การเล่นพิณ:

Requirements:
1. ใช้ PyTorch + Transformers library
2. รองรับ multi-GPU training
3. มี checkpointing และ resuming
4. Logging ด้วย Weights & Biases
5. Auto-evaluation ทุก N steps
6. Early stopping mechanism

Script ต้องมีส่วนประกอบ:
1. Data loading และ preprocessing
2. Model initialization
3. Training loop with validation
4. Checkpoint management
5. Metrics logging
6. Audio sample generation ระหว่าง training

Include:
- Progress bars
- Error handling
- Memory optimization
- Mixed precision training (FP16)
- Gradient clipping
- Learning rate scheduling

ตัวอย่าง command line usage:
python train_pin_model.py \\\n  --data_dir ./pin_dataset \\\n  --output_dir ./checkpoints \\\n  --model_name musicgen-pin-isan \\\n  --batch_size 4 \\\n  --num_epochs 50 \\\n  --learning_rate 5e-5 \\\n  --wandb_project isan-pin-ai

สร้าง complete, production-ready code"""
}

# Inference & Generation Prompts
INFERENCE_PROMPTS = {
    "generation": """คุณเป็น AI Generator สำหรับเสียงพิณอีสาน

ใช้โมเดลที่ train แล้วในการสร้างเสียงพิณตามคำสั่งของผู้ใช้

Input Format:
{
  "command": "user's natural language request",
  "parameters": {
    "style": "optional",
    "tempo": "optional",
    "mood": "optional",
    "duration": "optional (seconds)",
    "key": "optional",
    "reference_audio": "optional file path"
  }
}

คำสั่งที่รองรับ:
1. "สร้างเสียงพิณแบบหมอลำเพลินช้าๆ เศร้าๆ นาน 30 วินาที"
2. "เล่นพิณสไตล์อีสานเหนือ จังหวะเร็ว สนุกสนาน"
3. "ทำเสียงพิณโรแมนติก เหมาะกับงานแต่งงาน"
4. "สร้างเสียงพิณประกอบการเล่าเรื่อง ดราม่าติก"
5. "เปลี่ยนสไตล์เสียงพิณนี้ให้เป็นแบบซิ่ง" + [ไฟล์ต้นฉบับ]

Processing Steps:
1. Parse user command
2. Extract parameters
3. Generate text prompt สำหรับโมเดล
4. Run inference
5. Post-process audio (normalize, fade in/out)
6. Return audio file + metadata

Output Format:
{
  "audio_file": "generated_pin_001.wav",
  "duration": 30.5,
  "generated_prompt": "Generate Lam Plearn Isan Pin...",
  "parameters_used": {...},
  "generation_time": 5.2,
  "quality_score": 8.5
}

ตอบกลับในรูปแบบ conversational และให้คำแนะนำเพิ่มเติมถ้าจำเป็น""",

    "interactive": """สร้าง Interactive Generation System สำหรับเสียงพิณ:

Features:
1. Real-time generation (streaming)
2. Progressive refinement
3. User feedback loop
4. Style mixing
5. Variation generation

Conversation Flow:
User: "อยากได้เสียงพิณแบบเศร้าๆ"
AI: "เข้าใจค่ะ คุณต้องการแบบหมอลำเพลินใช่ไหมคะ? \\\n     และต้องการความยาวประมาณเท่าไหร่คะ?"

User: "ใช่ ประมาณ 1 นาที"
AI: [สร้างเสียง] "ฟังตัวอย่างนี้ค่ะ [audio] \\\n     ถ้าต้องการปรับแต่ง บอกได้เลยนะคะ"

User: "ช้าลงอีกนิดได้ไหม"
AI: [ปรับแต่ง] "ได้ค่ะ นี่เป็นเวอร์ชันช้าลง [audio]\\\n         ชอบแบบนี้ไหมคะ?"

Interactive Controls:
- ปรับ tempo: "เร็วขึ้น/ช้าลง"
- เปลี่ยน mood: "เศร้าขึ้น/สนุกขึ้น"
- เพิ่ม variation: "เล่นอีกแบบ"
- Mix styles: "ผสมกับแบบซิ่ง"
- Add instruments: "เพิ่มเสียงแคน"

System ต้อง:
- จำบริบทของการสนทนา
- เรียนรู้จาก user preferences
- Suggest improvements
- Explain choices

สร้าง complete chatbot interface"""
}

# Style Transfer Prompts
STYLE_TRANSFER_PROMPTS = {
    "transfer_system": """สร้างระบบ Style Transfer สำหรับเสียงพิณ:

Task: แปลงเสียงพิณจากสไตล์หนึ่งไปอีกสไตล์หนึ่ง โดยคงเนื้อหาทำนองไว้

Supported Transformations:
1. หมอลำเพลิน → หมอลำซิ่ง
2. ช้า → เร็ว (และในทางกลับกัน)
3. เศร้า → สนุกสนาน
4. อีสานเหนือ → อีสานใต้
5. โซโล → ประสานเสียง

Input:
- Source audio file (WAV/MP3)
- Target style description
- Preservation level (0.0-1.0): ระดับการรักษาลักษณะต้นฉบับ

Processing Pipeline:
1. Extract content features (melody, rhythm)
2. Extract style features (timbre, articulation)
3. Separate content and style
4. Apply target style
5. Reconstruct audio
6. Post-process and blend

Parameters:
{
  "content_weight": 0.7,
  "style_weight": 0.3,
  "smoothness": 0.5,
  "tempo_adjustment": 1.0,
  "key_shift": 0
}

Output:
- Transformed audio
- Side-by-side comparison
- Difference visualization
- Quality metrics

Example Usage:
input_audio = "lam_plearn_slow.wav"
target_style = "lam_sing_fast"
result = transform_pin_style(input_audio, target_style, \\\n                             content_weight=0.7)

สร้าง implementation ด้วย neural style transfer techniques"""
}

# Evaluation Prompts
EVALUATION_PROMPTS = {
    "quality_assessment": """สร้างระบบประเมินคุณภาพเสียงพิณที่ AI สร้างขึ้น:

Evaluation Dimensions:

1. Technical Quality (0-10):
   - Audio fidelity
   - Noise level
   - Dynamic range
   - Frequency balance
   - Artifacts detection

2. Musical Quality (0-10):
   - Melody coherence
   - Rhythm consistency
   - Harmonic correctness
   - Phrasing naturalness
   - Overall musicality

3. Style Authenticity (0-10):
   - Traditional accuracy
   - Regional characteristics
   - Performance techniques
   - Cultural appropriateness
   - Expert validation

4. Emotional Impact (0-10):
   - Mood conveyance
   - Storytelling ability
   - Listener engagement
   - Emotional resonance

Evaluation Methods:
1. Objective Metrics:
   - FAD (Frechet Audio Distance)
   - MCD (Mel Cepstral Distortion)
   - SNR (Signal-to-Noise Ratio)
   - Perceptual metrics

2. Subjective Testing:
   - A/B comparison with real recordings
   - Expert musician ratings
   - Listener surveys
   - Turing test (can people distinguish AI from human?)

Generate Detailed Report:
{
  "overall_score": 8.5,
  "technical_quality": 9.0,
  "musical_quality": 8.5,
  "style_authenticity": 8.0,
  "emotional_impact": 8.5,
  "strengths": ["Natural timbre", "Good rhythm"],
  "weaknesses": ["Occasional glitches", "Less expressive"],
  "recommendations": ["Improve vibrato", "Add more dynamics"],
  "comparison_with_human": {
    "similarity": 0.85,
    "distinguishable": "Sometimes",
    "preferred_over_human": "40% of listeners"
  }
}

สร้าง automated evaluation pipeline""",

    "performance_analysis": """วิเคราะห์ประสิทธิภาพของโมเดล AI เล่นพิณ:

Analysis Areas:

1. Generation Quality vs Training Data Size:
   - Plot: Number of training samples vs Quality metrics
   - Find optimal dataset size
   - Identify overfitting/underfitting

2. Style Coverage:
   - Which styles generate well?
   - Which styles need more data?
   - Style confusion matrix

3. Tempo and Key Accuracy:
   - Requested vs Generated tempo distribution
   - Key accuracy percentage
   - Tempo drift over time

4. Diversity Analysis:
   - How diverse are the generations?
   - Measure variation within same prompt
   - Compare to human variation

5. Failure Cases:
   - Collect and categorize failure modes
   - Identify patterns in failures
   - Suggest improvements

6. Inference Performance:
   - Generation time vs audio length
   - Memory usage
   - GPU utilization
   - Optimization opportunities

Visualization Requirements:
- Interactive dashboards
- Audio waveform comparisons
- Spectrogram comparisons
- Statistical distributions
- Timeline of improvements

Tools:
- Use Weights & Biases for tracking
- TensorBoard for visualization
- Custom analysis scripts

Generate comprehensive analysis report with visualizations"""
}

# Application Development Prompts
APP_PROMPTS = {
    "web_app": """สร้าง Web Application สำหรับสร้างเสียงพิณด้วย AI:

Application Name: "พิณAI - Isan Pin Generator"

Features:
1. Homepage:
   - Simple text input: "บรรยายเสียงพิณที่ต้องการ"
   - Quick style buttons: [เพลิน][ซิ่ง][กลอน][ตัด]
   - Example prompts
   - Audio player with waveform

2. Advanced Settings:
   - Duration slider (15s - 5min)
   - Tempo control (60-160 BPM)
   - Mood selector
   - Key selector
   - Region preference

3. Style Transfer Page:
   - Upload audio file
   - Select target style
   - Preview before/after
   - Download result

4. Gallery:
   - Community generated sounds
   - Featured creations
   - Like and share

5. Learning Center:
   - About Isan music
   - Pin history and culture
   - How the AI works
   - Tutorial videos

Tech Stack:
- Frontend: React + Tailwind CSS
- Backend: FastAPI
- Model Serving: Hugging Face Inference API
- Database: PostgreSQL
- Storage: AWS S3
- Authentication: Auth0

API Endpoints:
POST /api/generate
POST /api/transfer
GET /api/gallery
GET /api/status/{job_id}

User Experience:
- Fast response time (<10s for 30s audio)
- Progress indicators
- Real-time preview
- Mobile responsive
- Accessibility features

สร้าง complete fullstack application""",

    "mobile_app": """สร้าง Mobile Application "พิณAI Mobile":

Platform: iOS และ Android (React Native)

Core Features:

1. Quick Generate:
   - Voice command: "สร้างเสียงพิณแบบเศร้าๆ"
   - One-tap generation
   - Preset styles

2. Record & Transform:
   - Record your humming
   - Convert to Pin style
   - Share with friends

3. Practice Mode:
   - Learn Pin patterns
   - Play-along tracks
   - Progress tracking

4. Offline Mode:
   - Download generated sounds
   - Local playback
   - Sync when online

5. Social Features:
   - Share creations
   - Collaborate with others
   - Join challenges

UI/UX Design:
- Bottom tab navigation
- Swipe gestures
- Dark/Light themes
- Thai language interface
- Isaan cultural elements in design

Technical Implementation:
- React Native
- Redux for state management
- React Native Sound for audio
- AsyncStorage for offline data
- Push notifications
- In-app purchases (premium features)

Premium Features:
- Longer generation time
- More style options
- High quality export
- Remove watermark
- Priority processing

สร้าง complete mobile app with beautiful UI"""
}

# Documentation Prompts
DOCUMENTATION_PROMPTS = {
    "technical_docs": """สร้าง comprehensive technical documentation สำหรับโครงการ AI เล่นพิณ:

Documentation Structure:

1. README.md:
   - Project overview
   - Quick start guide
   - Installation instructions
   - Basic usage examples
   - Links to detailed docs

2. docs/architecture.md:
   - System architecture diagram
   - Model architecture
   - Data pipeline
   - Infrastructure

3. docs/training.md:
   - Dataset preparation
   - Training procedure
   - Hyperparameter tuning
   - Evaluation methods

4. docs/api_reference.md:
   - All API endpoints
   - Request/response formats
   - Authentication
   - Rate limits
   - Error codes

5. docs/deployment.md:
   - Deployment options
   - Docker setup
   - Cloud deployment (AWS/GCP)
   - Monitoring and logging

6. docs/contributing.md:
   - How to contribute
   - Code style guide
   - Pull request process
   - Testing requirements

7. docs/faq.md:
   - Common questions
   - Troubleshooting
   - Performance tips

8. docs/cultural_notes.md:
   - About Isan music
   - Pin playing techniques
   - Cultural sensitivity
   - Proper usage

Writing Style:
- Clear and concise
- Code examples for everything
- Visual diagrams
- Both Thai and English
- Beginner-friendly

Tools:
- Use MkDocs or Docusaurus
- Auto-generate API docs
- Include video tutorials
- Interactive examples

สร้าง complete, professional documentation"""
}

# Research Prompts
RESEARCH_PROMPTS = {
    "experiments": """ออกแบบ research experiments เพื่อปรับปรุงโมเดล:

Experiment 1: Data Augmentation Impact
Hypothesis: การเพิ่ม augmentation จะเพิ่มความหลากหลายโดยไม่เสียคุณภาพ

Variables:
- Control: ไม่มี augmentation
- Test groups:
  A) Pitch shift only
  B) Time stretch only  
  C) Background noise only
  D) All augmentation combined

Metrics:
- Generation diversity (measured by ...)
- Audio quality (FAD score)
- Style accuracy

---

Experiment 2: Model Size vs Performance
Hypothesis: โมเดลขนาดใหญ่ไม่จำเป็นต้องดีกว่าเสมอสำหรับ niche domain

Compare:
- MusicGen Small (300M params)
- MusicGen Medium (1.5B params)
- MusicGen Large (3.3B params)

Measure:
- Quality metrics
- Inference speed
- Memory usage
- Training time

---

Experiment 3: Fine-tuning Strategies
Compare different fine-tuning approaches:

A) Full fine-tuning
B) LoRA (Low-Rank Adaptation)
C) Prefix tuning
D) Adapter layers

Measure:
- Final model quality
- Training efficiency
- Model size
- Adaptation speed

---

Experiment 4: Prompt Engineering
Test different prompt formats:

A) Simple: "Lam Plearn slow sad"
B) Detailed: "Generate Lam Plearn style..."
C) Structured: JSON format
D) Audio conditioning: text + reference audio

Measure quality and controllability

สร้าง experiment tracking system และ analyze results"""
}

# Example Usage Functions
def create_generation_prompt(user_input, style=None, tempo=None, mood=None):
    """Create a generation prompt from user input"""
    prompt = f"""Generate Isan Pin music based on user request: {user_input}
    
Parameters:"""
    
    if style:
        prompt += f"\\n- Style: {style}"
    if tempo:
        prompt += f"\\n- Tempo: {tempo} BPM"
    if mood:
        prompt += f"\\n- Mood: {mood}"
    
    prompt += """\\n\\nOutput format: Audio file (WAV, 32kHz, mono)
Duration: 30-120 seconds
Key: Auto-detect appropriate key
Time signature: 4/4 (unless specified)"""
    
    return prompt

def parse_user_request(user_input):
    """Parse Thai user request and extract parameters"""
    
    # Keyword mappings for Thai terms
    style_keywords = {
        'เพลิน': 'lam_plearn',
        'ซิ่ง': 'lam_sing', 
        'กลอน': 'lam_klorn',
        'ตัด': 'lam_tad',
        'พื้น': 'lam_puen'
    }
    
    tempo_keywords = {
        'ช้า': 'slow',
        'เร็ว': 'fast',
        'ปานกลาง': 'medium'
    }
    
    mood_keywords = {
        'เศร้า': 'sad',
        'สนุก': 'fun',
        'โรแมนติก': 'romantic',
        'ตื่นเต้น': 'exciting'
    }
    
    # Extract parameters
    parameters = {
        'style': None,
        'tempo': None, 
        'mood': None
    }
    
    for thai, english in style_keywords.items():
        if thai in user_input:
            parameters['style'] = english
            break
            
    for thai, english in tempo_keywords.items():
        if thai in user_input:
            parameters['tempo'] = english
            break
            
    for thai, english in mood_keywords.items():
        if thai in user_input:
            parameters['mood'] = english
            break
    
    return parameters

if __name__ == "__main__":
    # Test the prompts
    print("=== Isan Pin AI Prompts Collection ===\\n")
    
    print("Data Collection Prompts:")
    for key, prompt in DATA_COLLECTION_PROMPTS.items():
        print(f"\\n{key.upper()}:")
        print(prompt[:200] + "...")
    
    print("\\n\\n=== Example Usage ===")
    user_request = "สร้างเสียงพิณแบบเพลินๆ ช้าๆ เศร้าๆ"
    params = parse_user_request(user_request)
    print(f"\\nUser request: {user_request}")
    print(f"Parsed parameters: {params}")
    
    gen_prompt = create_generation_prompt(user_request, **params)
    print(f"\\nGenerated prompt: {gen_prompt[:300]}...")