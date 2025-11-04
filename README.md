# User Guide Summarization App �

Transform your user guide documentation into organized, digestible summaries using Azure OpenAI and Streamlit.

## Features ✨

- **Multiple Input Methods**: Upload files, paste text, or load sample user guides
- **🌐 Multi-Language Support**: Generate summaries and get answers in 10 different languages
- **Customizable Summaries**: Choose from concise, detailed, or action-focused styles
- **Advanced Settings**: Adjust output length and creativity levels
- **Download & Share**: Export summaries as text files
- **Real-time Statistics**: Track compression ratios and word counts
- **🤖 Smart Q&A Chatbot**: Interactive chatbot with automatic language detection
- **🔍 Auto-Language Detection**: Chatbot automatically detects and responds in question language
- **Chat History**: Persistent conversation history with export functionality
- **Suggested Questions**: Multi-language question suggestions for better interaction
- **Error Handling**: Graceful handling of API errors and invalid inputs

## Setup Instructions 🚀

### 1. Environment Setup

1. Create a `.env` file in the project root:

```bash
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
```

2. Install dependencies:

```bash
# Install UV package manager (if not already installed)
# Follow the official installation guide: https://docs.astral.sh/uv/getting-started/installation/

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate   # (Linux/macOS)
.venv\Scripts\activate      # (Windows)
```

### 2. Running the App

**First-time setup:**
```bash
# This will create uv.lock with exact dependency versions
uv sync
```

**Streamlit Web App:**

```bash
# With UV
uv run streamlit run main.py

# Or activate environment first
source .venv/bin/activate
streamlit run main.py
```

**Command Line Version:**

```bash
# With UV
uv run python backend_developer.py

# Or activate environment first
source .venv/bin/activate
python backend_developer.py
```

**Run Tests:**

```bash
# With UV
uv run python test_app.py

# Or activate environment first
source .venv/bin/activate
python test_app.py
```

## Usage Guide 📖

### Web Interface (app.py)

#### User Guide Summary Tab

1. **Configure Settings**: Use the sidebar to choose summary style and adjust parameters
2. **Input Method**: Choose from:
   - Upload a `.txt` file
   - Paste text directly
   - Load the sample user guide
3. **Generate Summary**: Click "Generate Summary" to process your user guide document
4. **Download Results**: Export your summary as a text file

#### Q&A Chatbot Tab

1. **Smart Interactive Chat**: Ask questions in any language - the bot automatically detects and responds in the same language
2. **Multi-Language Suggestions**: Click on suggested questions in different languages
3. **Automatic Language Detection**: No need to manually select language for questions
4. **Chat History**: View previous questions and answers in their original languages
5. **Export Chat**: Download the entire multilingual chat session as a text file
6. **Clear Chat**: Reset the conversation history when needed

### Configuration Options

- **Language Selection** 🌐:

  - English, Spanish, French, German, Italian
  - Portuguese, Japanese, Chinese (Simplified), Korean, Arabic

- **Summary Styles**:

  - `concise`: Brief bullet points with key features and information
  - `detailed`: Comprehensive summary with all sections and procedures
  - `action-focused`: Emphasis on step-by-step instructions and guidelines

- **Advanced Settings**:
  - `Max Output Length`: Control summary length (150-1000 tokens)
  - `Creativity`: Adjust response variability (0.0-1.0)

## Test Cases 🧪

The app includes comprehensive testing covering:

- **TC_01**: Short user guide document summarization
- **TC_02**: Long multi-section guide handling
- **TC_03**: Empty input validation
- **TC_04**: Authentication error handling
- **TC_05**: Chatbot Q&A functionality
- **TC_06**: Empty question handling
- **TC_07**: Multi-language summary generation
- **TC_08**: Multi-language Q&A responses
- **TC_09**: Automatic language detection accuracy
- **TC_10**: Auto-language Q&A responses

Run tests with: `uv run python test_app.py`

## System Architecture 🏗️

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Module  │ -> │  Azure OpenAI   │ -> │  Output Module  │
│                 │    │   API Wrapper   │    │                 │
│ - File Upload   │    │ - Prompt Eng.   │    │ - Display       │
│ - Text Input    │    │ - Error Handle  │    │ - Download      │
│ - Sample Data   │    │ - Response Proc │    │ - Statistics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Error Handling 🛡️

- **Missing Credentials**: Clear error messages with setup instructions
- **Empty Input**: Helpful prompts to retry with content
- **API Errors**: Graceful degradation with user-friendly messages
- **File Upload Issues**: Validation and format checking

## File Structure 📁

```
├── main.py               # Main Streamlit application
├── backend_developer.py   # CLI version
├── test_app.py           # Test suite
├── pyproject.toml        # Project configuration and dependencies
├── .python-version       # Python version for UV
├── .env                 # Environment variables (create this)
├── data/
│   └── user_guide_sample.txt  # Sample data
└── README.md            # This file
```

## Dependencies 📦

This project uses [UV](https://docs.astral.sh/uv/) for dependency management. Dependencies are defined in `pyproject.toml`:

- `streamlit>=1.28.0` - Web interface
- `openai>=1.0.0` - Azure OpenAI integration
- `python-dotenv>=1.0.0` - Environment variable management
- Additional ML and NLP libraries for advanced features

Install all dependencies with: `uv sync`

## Troubleshooting 🔧

**Common Issues:**

1. **"Missing credentials" error**: Ensure `.env` file exists with correct variables
2. **Import errors**: Run `uv sync` to install dependencies
3. **File not found**: Check that `data/user_guide_sample.txt` exists
4. **Streamlit not starting**: Verify installation with `uv run streamlit --version`
5. **UV not found**: Install UV following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/)

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License 📄

This project is open source and available under the MIT License.
