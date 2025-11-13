# Isan Pin AI - Traditional Thai Music Generation System

A comprehensive AI system for generating authentic traditional Isan Pin music from Northeast Thailand using deep learning and cultural preservation techniques.

## 🎵 Overview

Isan Pin AI is a sophisticated system that combines:
- **Cultural Preservation**: Maintains authentic Thai musical traditions
- **Advanced AI**: Uses Facebook MusicGen fine-tuned on traditional Isan Pin music
- **Style Classification**: CNN-based recognition of 5 traditional styles
- **Quality Assessment**: Multi-dimensional evaluation with cultural authenticity
- **User-Friendly Interface**: Both command-line and web interfaces

## 🎯 Key Features

### Traditional Music Styles Supported
- **Lam Plearn** (หมอลำเพลิน) - Slow, contemplative style (60-90 BPM)
- **Lam Sing** (หมอลำซิ่ง) - Fast, energetic style (120-160 BPM)  
- **Lam Klorn** (หมอลำกลอน) - Moderate, poetic style (80-120 BPM)
- **Lam Tad** (หมอลำตัด) - Medium-fast, narrative style (100-140 BPM)
- **Lam Puen** (หมอลำปึน) - Moderate, storytelling style (90-130 BPM)

### AI Capabilities
- 🎼 **Text-to-Music Generation**: Create music from Thai/English descriptions
- 🔄 **Style Transfer**: Convert between different Isan Pin styles
- 🔍 **Style Classification**: Automatically identify musical styles
- 📊 **Quality Assessment**: Evaluate cultural authenticity and audio quality
- 🌐 **Web Interface**: User-friendly Gradio and FastAPI interfaces

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

```bash
# Generate traditional Isan Pin music
python main.py generate --prompt "สร้างเสียงพิณแบบหมอลำเพลินช้าๆ เศร้าๆ" --style lam_plearn --duration 30

# Launch web interface
python main.py web --port 8000

# Train classifier on your data
python main.py train-classifier --data-dir ./dataset --output-dir ./models

# Evaluate generated music
python main.py evaluate --audio-file ./generated_music.wav --reference-style lam_plearn
```

### Python API

```python
from src import IsanPinInference, IsanPinEvaluator

# Initialize inference system
inference = IsanPinInference(
    classifier_path="./models/classifier.pth",
    generator_path="./models/generator.pth"
)

# Generate music
results = inference.generate_music(
    description="Traditional Isan Pin music with slow tempo and contemplative mood",
    style="lam_plearn",
    duration=30.0,
    num_samples=1
)

# Evaluate quality
evaluator = IsanPinEvaluator()
results = evaluator.evaluate_single_audio(
    audio="./generated_music.wav",
    reference_style="lam_plearn"
)
print(f"Cultural Authenticity Score: {results['cultural_authenticity']['cultural_authenticity_score']}")
```

## 🎨 Web Interface

Access the web interface at `http://localhost:8000` after running:

```bash
python main.py web --port 8000
```

### Features
- **Generate Music**: Create music from text descriptions
- **Style Transfer**: Apply different Isan Pin styles to existing audio
- **Evaluate Audio**: Assess quality and cultural authenticity
- **Download Results**: Export generated music in various formats

## 📁 Project Structure

```
isan-pin-ai/
├── src/                          # Source code
│   ├── config.py                # Configuration settings
│   ├── data/                    # Data processing modules
│   │   ├── collection.py       # Audio collection and classification
│   │   └── preprocessing.py    # Audio preprocessing and dataset creation
│   ├── models/                  # AI models
│   │   ├── classification.py      # CNN-based style classifier
│   │   └── musicgen.py        # MusicGen fine-tuning
│   ├── inference/               # Inference pipeline
│   │   └── inference.py       # High-level inference system
│   ├── evaluation/              # Quality assessment
│   │   └── evaluator.py       # Comprehensive evaluation system
│   └── web/                     # Web interfaces
│       └── app.py              # FastAPI + Gradio application
├── main.py                      # Command-line interface
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🎯 Detailed Usage Examples

### 1. Data Collection and Preprocessing

```bash
# Collect and classify raw audio files
python main.py collect-data \
    --input-dir ./raw_audio \
    --output-dir ./processed \
    --sample-rate 48000 \
    --classify
```

### 2. Training Models

```bash
# Train style classifier
python main.py train-classifier \
    --data-dir ./dataset \
    --output-dir ./models/classifier \
    --epochs 50 \
    --batch-size 16

# Fine-tune music generator
python main.py train-generator \
    --data-dir ./dataset \
    --output-dir ./models/generator \
    --base-model facebook/musicgen-small \
    --epochs 10 \
    --batch-size 4
```

### 3. Music Generation

```bash
# Generate with specific style
python main.py generate \
    --prompt "Traditional Isan Pin music for meditation and relaxation" \
    --style lam_plearn \
    --duration 60 \
    --output-file ./meditation_pin.wav

# Generate with mood
python main.py generate \
    --prompt "เสียงพิณแบบไทยดั้งเดิม ฟังเพลินๆ" \
    --style lam_sing \
    --mood joyful \
    --duration 45
```

### 4. Style Transfer

```bash
# Transfer to different style
python main.py transfer-style \
    --input-audio ./input_music.wav \
    --target-style lam_klorn \
    --output-file ./transferred_klorn.wav \
    --strength 0.8
```

### 5. Evaluation

```bash
# Evaluate single file
python main.py evaluate \
    --audio-file ./generated_music.wav \
    --reference-style lam_plearn \
    --detailed

# Evaluate dataset
python main.py evaluate-dataset \
    --data-dir ./test_dataset \
    --output-report ./evaluation_report.json
```

## 🔬 Evaluation Metrics

The system provides comprehensive evaluation:

### Basic Quality Metrics
- **RMS Energy**: Audio signal strength
- **Dynamic Range**: Variation in loudness
- **Zero Crossing Rate**: Rhythmic complexity
- **Quality Score**: Overall technical quality (0-1)

### Style Consistency
- **Tempo Consistency**: Alignment with style-specific tempo ranges
- **Spectral Consistency**: Matching spectral characteristics
- **MFCC Consistency**: Mel-frequency cepstral coefficients alignment

### Cultural Authenticity
- **Tempo Authenticity**: Traditional tempo ranges
- **Rhythmic Authenticity**: Cultural rhythmic patterns
- **Spectral Authenticity**: Traditional timbral characteristics

### Reference Comparison
- **Similarity Score**: Comparison with reference Isan Pin recordings
- **Classification Consistency**: Agreement with style classifier

## 🎛️ Configuration

Edit `src/config.py` to customize:

```python
# Audio settings
AUDIO_CONFIG = {
    "sample_rate": 48000,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
}

# Style definitions
STYLE_DEFINITIONS = {
    "lam_plearn": {
        "thai_name": "หมอลำเพลิน",
        "tempo_range": [60, 90],
        "mood": ["sad", "romantic", "contemplative"],
    }
    # ... more styles
}
```

## 🌐 API Usage

### FastAPI Endpoints

```bash
# Get available styles
curl http://localhost:8000/styles

# Generate music
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Traditional Isan Pin music",
    "style": "lam_plearn",
    "duration": 30,
    "num_samples": 1
  }'

# Upload and evaluate
curl -X POST http://localhost:8000/evaluate \
  -F "audio_file=@./music.wav" \
  -F "reference_style=lam_plearn"
```

### Gradio Interface

Access the user-friendly interface at `http://localhost:8000/gradio`

## 🧪 Advanced Features

### Batch Processing

```python
# Generate multiple samples
results = inference.batch_generate(
    descriptions=["slow meditation music", "fast dance music"],
    styles=["lam_plearn", "lam_sing"],
    num_samples_per_description=3,
    max_workers=4
)
```

### Style Interpolation

```python
# Create smooth transitions between styles
interpolations = inference.interpolate_styles(
    description="Traditional Isan Pin music",
    style1="lam_plearn",
    style2="lam_sing", 
    num_steps=5,
    duration=30.0
)
```

### Custom Training

```python
# Prepare dataset
dataset = IsanPinDataset(
    audio_paths=audio_files,
    descriptions=descriptions,
    preprocessor=AudioPreprocessor()
)

# Train classifier
classifier = IsanPinClassifier()
history = classifier.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=50,
    save_path="./models/classifier.pth"
)
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Audio Quality Issues**: Adjust temperature and guidance_scale parameters
3. **Style Misclassification**: Retrain classifier with more diverse data
4. **Slow Generation**: Use smaller model or reduce duration

### Performance Tips

- Use GPU for faster training and generation
- Batch process multiple files when possible
- Pre-compute reference features for faster evaluation
- Cache generated samples for repeated use

## 📈 Future Enhancements

- [ ] Real-time generation capabilities
- [ ] Mobile app development
- [ ] Integration with Thai music notation
- [ ] Collaborative filtering for recommendations
- [ ] Multi-instrument generation
- [ ] Live performance integration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Facebook MusicGen team for the base model
- Thai musicologists and cultural experts
- Isan Pin musicians for preserving traditional music
- Open source community for supporting tools

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Preserve Thai Culture Through AI** 🎵🇹🇭