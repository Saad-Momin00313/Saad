# Mindful Moments

A Streamlit-based mindfulness application that provides personalized meditation and mindfulness activities using Google's Gemini AI.

## Features

- 🧘 Personalized mindfulness activities based on mood and energy levels
- 📊 Progress tracking with interactive visualizations
- 🎯 Custom practice routines and goals
- 📈 Mood and energy level tracking
- ⚡ Quick stress relief practices
- 🌙 Sleep improvement recommendations

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd mindfull
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run main.py
```

## Usage

1. **Quick Practice**: Select your current mood, energy level, and available time to get a personalized mindfulness activity.
2. **Progress Tracking**: View your practice statistics and trends in the Progress tab.
3. **Settings**: Customize your practice goals and preferences in the Settings tab.

## Requirements

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- Google Generative AI
- Python-dotenv

## License

MIT License
