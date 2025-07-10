# Tamil-English YouTube Comment Sentiment Analysis Tool

A comprehensive sentiment analysis tool specifically designed for Tamil and Tamil-English code-mixed text from YouTube comments. This tool utilises Meta's Llama 3.2-1B-Instruct model to deliver accurate sentiment classification, along with interactive visualisations and batch processing capabilities.

## 🌟 Features

- **Multi-language Support**: Analyzes Tamil and Tamil-English code-mixed text
- **Advanced AI Model**: Uses Meta's Llama 3.2-1B-Instruct for accurate sentiment classification
- **YouTube Integration**: Extracts comments directly from YouTube videos via API
- **Interactive Mode**: User-friendly interface for real-time analysis
- **Batch Processing**: Analyze multiple videos simultaneously
- **Rich Visualizations**: 
  - Sentiment distribution charts (bar and pie charts)
  - Time-series sentiment trends
  - Engagement analysis (likes vs sentiment correlation)
  - Word clouds for each sentiment category
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs with 4-bit quantization
- **Export Functionality**: Save results to CSV files
- **Comprehensive Reports**: Detailed analysis reports with key insights

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- YouTube Data API v3 key
- Hugging Face account and token

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/tamil-yt-sentiment-analysis.git
cd tamil-yt-sentiment-analysis
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv tamil_sentiment_env

# Activate virtual environment
# On Windows:
tamil_sentiment_env\Scripts\activate
# On macOS/Linux:
source tamil_sentiment_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (check https://pytorch.org for GPU-specific installation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install --upgrade transformers accelerate bitsandbytes
pip install google-api-python-client pandas matplotlib seaborn wordcloud
pip install huggingface_hub datasets evaluate
pip install textblob langdetect
```

### Step 4: Set Up API Keys

#### YouTube Data API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Copy the API key

#### Hugging Face Token
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up/Sign in to your account
3. Go to Settings → Access Tokens
4. Create a new token with read permissions
5. Copy the token

#### Environment Setup
Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
HF_TOKEN=your_hugging_face_token_here
```

Or set environment variables:

```bash
export YOUTUBE_API_KEY="your_youtube_api_key_here"
export HF_TOKEN="your_hugging_face_token_here"
```

## 🚀 Usage

### Interactive Mode

```python
python tamil_yt_comment_analysis.py
```

Then run the interactive function:

```python
interactive_analysis()
```

### Programmatic Usage

```python
from tamil_yt_comment_analysis import analyze_youtube_video, SentimentVisualizer

# Analyze a single video
video_url = "https://youtu.be/VIDEO_ID"
df, video_info = analyze_youtube_video(video_url, max_comments=100)

# Create visualizations
visualizer = SentimentVisualizer(df)
visualizer.plot_sentiment_distribution()
visualizer.generate_summary_report(video_info)
```

### Batch Processing

```python
from tamil_yt_comment_analysis import batch_analyze_videos

video_urls = [
    "https://youtu.be/VIDEO_ID_1",
    "https://youtu.be/VIDEO_ID_2",
    "https://youtu.be/VIDEO_ID_3"
]

batch_results = batch_analyze_videos(video_urls, max_comments_per_video=50)
```

### Custom Text Analysis

```python
from tamil_yt_comment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.classify_sentiment("இந்த படம் மிகவும் அருமையாக இருந்தது!")
print(f"Sentiment: {sentiment}")  # Output: Positive
```

## 📊 Sample Output

### Sentiment Distribution
- **Positive**: 45.2% (127 comments)
- **Negative**: 23.8% (67 comments)  
- **Neutral**: 31.0% (87 comments)

### Key Insights
- Most liked positive comment: "Super movie! Acting vera level 🔥"
- Average engagement by sentiment:
  - Positive: 15.3 likes
  - Negative: 8.7 likes
  - Neutral: 5.2 likes

## 🏗️ Project Structure

```
tamil-yt-sentiment-analysis/
├── tamil_yt_comment_analysis.py    # Main application file
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore file
├── LICENSE                         # License file
└── examples/                       # Example usage scripts
    ├── basic_analysis.py
    ├── batch_processing.py
    └── custom_text_analysis.py
```

## 🔧 Configuration

The tool uses a `Config` class for easy customization:

```python
class Config:
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    MAX_NEW_TOKENS = 10
    TEMPERATURE = 0.1
    TOP_P = 0.9
    MAX_COMMENTS_TO_FETCH = 100
    COMMENTS_PER_PAGE = 50
    FIGSIZE = (12, 8)
    COLORS = ['#ff9999', '#66b3ff', '#99ff99']
```

## 📋 Requirements

### Python Dependencies
- transformers>=4.21.0
- torch>=1.12.0
- accelerate>=0.20.0
- bitsandbytes>=0.39.0
- google-api-python-client>=2.0.0
- pandas>=1.5.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- wordcloud>=1.9.0
- huggingface_hub>=0.15.0
- datasets>=2.0.0
- evaluate>=0.4.0
- textblob>=0.17.0
- langdetect>=1.0.9

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only processing
- **Recommended**: 16GB+ RAM, CUDA-compatible GPU with 8GB+ VRAM
- **Storage**: 5GB+ free space for model downloads

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/tamil-yt-sentiment-analysis.git
cd tamil-yt-sentiment-analysis

# Add upstream remote
git remote add upstream https://github.com/originalowner/tamil-yt-sentiment-analysis.git

# Create development branch
git checkout -b develop

# Install development dependencies
pip install -e .
pip install pytest black flake8 mypy
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `MAX_COMMENTS_TO_FETCH` in config
   - Enable 4-bit quantization (already enabled by default)
   - Use CPU-only mode if necessary

2. **YouTube API Quota Exceeded**
   - Wait for quota reset (daily limit)
   - Use multiple API keys if needed
   - Reduce `COMMENTS_PER_PAGE` for slower processing

3. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face token permissions
   - Try downloading model manually

### Performance Tips

- Use GPU acceleration for faster processing
- Process comments in batches for large datasets
- Cache model locally after first download
- Use appropriate quantization settings


## 🙏 Acknowledgments

- Meta AI for the Llama 3.2-1B-Instruct model
- Hugging Face for the transformers library
- Google for the YouTube Data API
- Tamil NLP community for inspiration

## 📈 Roadmap

- [ ] Support for more Indian languages .. [ telugu is ready ](https://github.com/Rishpraveen/Telugu_youtube_sentiment_analysis)
- [ ] Real-time sentiment monitoring
- [ ] Advanced emotion detection
- [ ] Web interface development
- [ ] Docker containerization
- [ ] API endpoint creation
- [ ] Mobile app development

## 🏷️ Version History

- **v1.0.0**: Initial release with basic sentiment analysis
- **v1.1.0**: Added batch processing and enhanced visualisations
- **v1.2.0**: Improved model accuracy and added GPU support

---

## 📱 Quick Start Example

```python
# Quick start - analyse a Tamil movie trailer's comments
from tamil_yt_comment_analysis import analyze_youtube_video, SentimentVisualizer

# Replace with actual YouTube URL
video_url = "https://youtu.be/dQw4w9WgXcQ"
df, video_info = analyze_youtube_video(video_url, max_comments=50)

# Generate visualizations
visualizer = SentimentVisualizer(df)
visualizer.plot_sentiment_distribution()
visualizer.generate_summary_report(video_info)

print("Analysis complete! Check the generated charts and CSV file.")
```

Made with ❤️ for the Tamilians
