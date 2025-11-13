# พิณAI - Isan Pin AI Prompts Collection

## ภาษาไทย | [English](#english-version)

คอลเล็กชันของ prompts สำหรับการพัฒนา AI สร้างเสียงพิณอีสาน รวมถึงการเก็บข้อมูล, การสอนโมเดล, และการสร้างเสียง

## 📋 สารบัญ

1. [การเก็บและจัดการข้อมูล](#การเก็บและจัดการข้อมูล)
2. [การสอนโมเดล](#การสอนโมเดล)
3. [การสร้างเสียง](#การสร้างเสียง)
4. [การถ่ายโอนสไตล์](#การถ่ายโอนสไตล์)
5. [การประเมินผล](#การประเมินผล)
6. [การพัฒนาแอปพลิเคชัน](#การพัฒนาแอปพลิเคชัน)
7. [การวิจัยและปรับปรุง](#การวิจัยและปรับปรุง)

## 🔧 การเก็บและจัดการข้อมูล

### การจัดหมวดหมู่ไฟล์เสียง

```python
from pin_ai_prompts import DATA_COLLECTION_PROMPTS

classification_prompt = DATA_COLLECTION_PROMPTS["audio_classification"]
print(classification_prompt)
```

ระบุรายละเอียดสำหรับการวิเคราะห์ไฟล์เสียง:
- สไตล์การเล่น: หมอลำเพลิน, หมอลำซิ่ง, หมอลำกลอน, ลำตัด, ลำพื้น
- Tempo: ช้า (60-80 BPM), ปานกลาง (80-120 BPM), เร็ว (120-160 BPM)
- อารมณ์: เศร้า, สนุกสนาน, โรแมนติก, ตื่นเต้น, เฉยๆ
- ภูมิภาค: อีสานเหนือ, อีสานใต้, อีสานกลาง
- เทคนิคพิเศษ: การสั่น, การเล่นเร็ว, การเด้ง, การประสาน

### การสร้าง Dataset

```python
dataset_prompt = DATA_COLLECTION_PROMPTS["dataset_creation"]
```

สร้าง training dataset ที่มี:
- จำนวนตัวอย่าง: 1000 ไฟล์
- ความยาว: 30-180 วินาที
- Sample rate: 48kHz, 24-bit

## 🎯 การสอนโมเดล

### Fine-tuning MusicGen

```python
from pin_ai_prompts import TRAINING_PROMPTS

finetune_config = TRAINING_PROMPTS["musicgen_finetune"]
print(finetune_config)
```

รายละเอียดการตั้งค่า:
- Base Model: facebook/musicgen-small
- Learning Rate: 5e-5
- Batch Size: 4
- Epochs: 50
- Dataset: 800 training, 150 validation, 50 test

### Training Script

```python
training_script = TRAINING_PROMPTS["training_script"]
```

สคริปต์สำหรับการสอนที่รวม:
- Multi-GPU training
- Checkpoint management
- Weights & Biases logging
- Auto-evaluation
- Mixed precision training

## 🎵 การสร้างเสียง

### สร้างเสียงใหม่

```python
from pin_ai_prompts import INFERENCE_PROMPTS

generation_prompt = INFERENCE_PROMPTS["generation"]
```

รองรับคำสั่งต่างๆ:
- "สร้างเสียงพิณแบบหมอลำเพลินช้าๆ เศร้าๆ นาน 30 วินาที"
- "เล่นพิณสไตล์อีสานเหนือ จังหวะเร็ว สนุกสนาน"
- "ทำเสียงพิณโรแมนติก เหมาะกับงานแต่งงาน"

### Interactive Generation

```python
interactive_system = INFERENCE_PROMPTS["interactive"]
```

ระบบสนทนาแบบโต้ตอบ:
- Real-time generation
- Progressive refinement
- User feedback loop
- Style mixing
- Variation generation

## 🔄 การถ่ายโอนสไตล์

```python
from pin_ai_prompts import STYLE_TRANSFER_PROMPTS

transfer_system = STYLE_TRANSFER_PROMPTS["transfer_system"]
```

การแปลงสไตล์เสียงพิณ:
- หมอลำเพลิน → หมอลำซิ่ง
- ช้า → เร็ว
- เศร้า → สนุกสนาน
- อีสานเหนือ → อีสานใต้

## 📊 การประเมินผล

### ประเมินคุณภาพเสียง

```python
from pin_ai_prompts import EVALUATION_PROMPTS

quality_assessment = EVALUATION_PROMPTS["quality_assessment"]
```

ประเมินตามมิติต่างๆ:
- Technical Quality (0-10)
- Musical Quality (0-10)
- Style Authenticity (0-10)
- Emotional Impact (0-10)

### วิเคราะห์ประสิทธิภาพโมเดล

```python
performance_analysis = EVALUATION_PROMPTS["performance_analysis"]
```

การวิเคราะห์:
- Generation Quality vs Training Data Size
- Style Coverage Analysis
- Tempo and Key Accuracy
- Diversity Analysis
- Failure Cases

## 📱 การพัฒนาแอปพลิเคชัน

### Web Application

```python
from pin_ai_prompts import APP_PROMPTS

web_app_design = APP_PROMPTS["web_app"]
```

แอปพลิเคชัน "พิณAI - Isan Pin Generator" รวม:
- Homepage และ Quick generation
- Advanced settings
- Style transfer
- Gallery และ community features
- Learning center

### Mobile Application

```python
mobile_app_design = APP_PROMPTS["mobile_app"]
```

แอปพลิเคชัน "พิณAI Mobile":
- Quick generate with voice command
- Record & transform
- Practice mode
- Offline mode
- Social features

## 🔬 การวิจัยและปรับปรุง

### การทดลองวิจัย

```python
from pin_ai_prompts import RESEARCH_PROMPTS

research_experiments = RESEARCH_PROMPTS["experiments"]
```

การทดลองต่างๆ:
- Data Augmentation Impact
- Model Size vs Performance
- Fine-tuning Strategies
- Prompt Engineering

## 📝 ตัวอย่างการใช้งาน

```python
from pin_ai_prompts import parse_user_request, create_generation_prompt

# แยกคำขอของผู้ใช้
user_request = "สร้างเสียงพิณแบบเพลินๆ ช้าๆ เศร้าๆ"
params = parse_user_request(user_request)
print(f"Parsed parameters: {params}")

# สร้าง prompt สำหรับ generation
gen_prompt = create_generation_prompt(user_request, **params)
print(f"Generated prompt: {gen_prompt}")
```

## 🚀 การเริ่มต้นใช้งาน

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. นำเข้า prompts:
```python
from pin_ai_prompts import *
```

3. ใช้งานตามตัวอย่างด้านบน

## 📄 ไฟล์ในโปรเจกต์

- `pin_ai_prompts.py` - คอลเล็กชัน prompts หลัก
- `README.md` - ไฟล์นี้

---

## English Version

## 🎯 Overview

This is a comprehensive collection of prompts for developing AI that generates traditional Isan Pin music from Northeast Thailand. The prompts cover the entire development pipeline from data collection to model evaluation.

## 📋 Table of Contents

1. [Data Collection & Preparation](#data-collection--preparation)
2. [Model Training](#model-training)
3. [Inference & Generation](#inference--generation)
4. [Style Transfer](#style-transfer)
5. [Evaluation & Analysis](#evaluation--analysis)
6. [Application Development](#application-development)
7. [Research & Improvement](#research--improvement)

## 🔧 Data Collection & Preparation

### Audio Classification

```python
from pin_ai_prompts import DATA_COLLECTION_PROMPTS

classification_prompt = DATA_COLLECTION_PROMPTS["audio_classification"]
```

Analyzes audio files for:
- Playing styles: Lam Plearn, Lam Sing, Lam Klorn, Lam Tad, Lam Puen
- Tempo: Slow (60-80 BPM), Medium (80-120 BPM), Fast (120-160 BPM)
- Mood: Sad, Fun, Romantic, Exciting, Neutral
- Region: Northern Isan, Southern Isan, Central Isan
- Special techniques: Vibrato, Fast plucking, Bouncing, Harmonization

### Dataset Creation

```python
dataset_prompt = DATA_COLLECTION_PROMPTS["dataset_creation"]
```

Creates training datasets with:
- 1000 audio samples
- Duration: 30-180 seconds
- Sample rate: 48kHz, 24-bit
- Metadata and descriptions

## 🎯 Model Training

### Fine-tuning MusicGen

```python
from pin_ai_prompts import TRAINING_PROMPTS

finetune_config = TRAINING_PROMPTS["musicgen_finetune"]
```

Training configuration:
- Base Model: facebook/musicgen-small
- Learning Rate: 5e-5
- Batch Size: 4
- Epochs: 50
- Dataset: 800 training, 150 validation, 50 test samples

### Training Script

```python
training_script = TRAINING_PROMPTS["training_script"]
```

Complete training script with:
- Multi-GPU training support
- Checkpoint management
- Weights & Biases logging
- Auto-evaluation
- Mixed precision training

## 🎵 Inference & Generation

### Generate New Audio

```python
from pin_ai_prompts import INFERENCE_PROMPTS

generation_prompt = INFERENCE_PROMPTS["generation"]
```

Supports commands like:
- "Create slow sad Isan Pin music for 30 seconds"
- "Generate fast fun Northern Isan Pin music"
- "Create romantic Pin music suitable for weddings"

### Interactive Generation

```python
interactive_system = INFERENCE_PROMPTS["interactive"]
```

Interactive conversation system with:
- Real-time generation
- Progressive refinement
- User feedback loop
- Style mixing
- Variation generation

## 🔄 Style Transfer

```python
from pin_ai_prompts import STYLE_TRANSFER_PROMPTS

transfer_system = STYLE_TRANSFER_PROMPTS["transfer_system"]
```

Style transfer capabilities:
- Lam Plearn → Lam Sing
- Slow → Fast
- Sad → Fun
- Northern Isan → Southern Isan

## 📊 Evaluation & Analysis

### Audio Quality Assessment

```python
from pin_ai_prompts import EVALUATION_PROMPTS

quality_assessment = EVALUATION_PROMPTS["quality_assessment"]
```

Evaluation dimensions:
- Technical Quality (0-10)
- Musical Quality (0-10)
- Style Authenticity (0-10)
- Emotional Impact (0-10)

### Model Performance Analysis

```python
performance_analysis = EVALUATION_PROMPTS["performance_analysis"]
```

Analysis includes:
- Generation Quality vs Training Data Size
- Style Coverage Analysis
- Tempo and Key Accuracy
- Diversity Analysis
- Failure Cases

## 📱 Application Development

### Web Application

```python
from pin_ai_prompts import APP_PROMPTS

web_app_design = APP_PROMPTS["web_app"]
```

"พิณAI - Isan Pin Generator" web app features:
- Homepage with quick generation
- Advanced settings
- Style transfer
- Gallery and community features
- Learning center

### Mobile Application

```python
mobile_app_design = APP_PROMPTS["mobile_app"]
```

"พิณAI Mobile" app features:
- Quick generate with voice command
- Record & transform
- Practice mode
- Offline mode
- Social features

## 🔬 Research & Improvement

### Research Experiments

```python
from pin_ai_prompts import RESEARCH_PROMPTS

research_experiments = RESEARCH_PROMPTS["experiments"]
```

Research experiments:
- Data Augmentation Impact
- Model Size vs Performance
- Fine-tuning Strategies
- Prompt Engineering

## 📝 Usage Examples

```python
from pin_ai_prompts import parse_user_request, create_generation_prompt

# Parse user request
user_request = "Create slow sad Lam Plearn style Pin music"
params = parse_user_request(user_request)
print(f"Parsed parameters: {params}")

# Create generation prompt
gen_prompt = create_generation_prompt(user_request, **params)
print(f"Generated prompt: {gen_prompt}")
```

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Import prompts:
```python
from pin_ai_prompts import *
```

3. Use according to examples above

## 📄 Files

- `pin_ai_prompts.py` - Main prompts collection
- `README.md` - This file

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Traditional Isan musicians and cultural experts
- Hugging Face for the MusicGen model
- Thai music researchers and ethnomusicologists

---

**สร้างโดย: AI Assistant สำหรับโครงการพิณAI**