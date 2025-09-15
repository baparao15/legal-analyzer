# Legal Document Analyzer 📜⚖️

An AI-powered legal document analyzer with a beautiful 3D UI that helps identify risky clauses in legal agreements and provides actionable recommendations using advanced NLP techniques and vector similarity search.

![Legal Document Analyzer](https://img.shields.io/badge/AI-Powered-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--DB-purple)

## 🌟 Features

- **🔍 AI-Powered Risk Detection**: Automatically identifies potentially risky clauses in legal documents
- **📊 3D Visualizations**: Beautiful, interactive 3D charts and graphs for risk assessment
- **💡 Smart Recommendations**: Get actionable advice for each identified risk
- **🎯 Risk Scoring**: Overall risk score calculation with detailed breakdown
- **📁 Document History**: Track all analyzed documents and their results
- **🔎 Semantic Search**: Find similar clauses using advanced AI embeddings
- **🎨 Modern UI**: Stunning 3D interface with glassmorphism effects

## 🚨 Risky Clauses Detected

The analyzer can detect various types of risky clauses including:

- **Unlimited Liability**: Clauses that make you responsible for unlimited damages
- **Auto-Renewal**: Contracts that automatically renew without notice
- **Unilateral Termination**: One-sided termination rights
- **IP Assignment**: Transfer of intellectual property rights
- **Non-Compete**: Restrictions on future employment
- **Binding Arbitration**: Mandatory arbitration clauses
- **Broad Confidentiality**: Overly restrictive confidentiality terms
- **Unfavorable Payment Terms**: Payment conditions that put you at risk
- **Warranty Disclaimers**: Lack of warranties or guarantees
- **Force Majeure**: Broad force majeure clauses

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd legal-document-analyzer
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model** (optional but recommended):
```bash
python -m spacy download en_core_web_sm
```

5. **Set up environment variables**:
```bash
# Create .env file and add your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## 🚀 Quick Start

### Method 1: Using Batch File (Windows)
```bash
# Double-click run.bat or execute in command prompt
run.bat
```

### Method 2: Direct Streamlit Command
```bash
streamlit run app.py
```

### Usage Steps
1. **Open your browser** and navigate to `http://localhost:8501`
2. **Upload a PDF** legal document using the sidebar
3. **Configure analysis settings**:
   - Enable "Use AI Enhanced Analysis" for GPT-4 powered analysis
   - Enable "Use Vector Database Context" for similarity search
4. **Click "Analyze Document"** to start the analysis
5. **Explore the results** in the different tabs:
   - 📋 **Analysis**: Overview and risk summary
   - 🔍 **Risk Details**: Detailed breakdown of each risky clause
   - 📄 **Document View**: Original document content

## 📁 Project Structure

```
legal-document-analyzer/
├── app.py                    # Main Streamlit application with 3D UI
├── legal_risk_analyzer.py    # Pattern-based risk detection engine
├── openai_legal_analyzer.py  # AI-powered risk analysis using GPT-4
├── pdf_vector_pipeline.py    # PDF processing and ChromaDB vector storage
├── requirements.txt          # Python dependencies
├── run.bat                  # Windows startup script
├── .env                     # Environment variables (API keys)
├── README.md                # This documentation
├── .streamlit/              # Streamlit configuration
│   └── config.toml
├── legal/                   # Sample legal documents
│   ├── *.pdf files
└── legal_vector_db/         # ChromaDB vector database (auto-generated)
    └── chroma.sqlite3
```

## 🔧 Configuration

### Analysis Settings

- **Analysis Depth**: Choose between Quick, Standard, or Deep analysis
- **Semantic Search**: Enable/disable AI-powered similarity search
- **Chunking Strategy**: Configure how documents are split for processing

### Customizing Risk Patterns

Edit `legal_risk_analyzer.py` to add or modify risk patterns:

```python
"your_custom_risk": {
    "patterns": [
        r"your regex pattern",
        r"another pattern"
    ],
    "risk_level": "high",  # or "medium", "low"
    "explanation": "Why this is risky",
    "recommendations": [
        "What to do about it",
        "Another recommendation"
    ]
}
```

## 💡 Tips for Best Results

1. **PDF Quality**: Ensure your PDFs are text-based (not scanned images)
2. **Document Length**: The analyzer works best with documents under 100 pages
3. **Language**: Currently optimized for English legal documents
4. **Format**: Standard legal document formats work best

## 🎨 UI Features

- **3D Card Effects**: Hover over cards to see 3D rotation effects
- **Glassmorphism**: Modern frosted glass design elements
- **Responsive Design**: Works on different screen sizes
- **Dark Theme**: Easy on the eyes with a professional dark theme
- **Smooth Animations**: Subtle animations for better user experience

## 🔒 Privacy & Security

- All document processing happens locally on your machine
- No data is sent to external servers
- Vector database is stored locally in `./legal_vector_db`
- Temporary files are automatically deleted after processing

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" error**:
   - Make sure you've activated your virtual environment
   - Run `pip install -r requirements.txt` again

2. **PDF extraction fails**:
   - Ensure the PDF is not password-protected
   - Check if the PDF contains actual text (not scanned images)

3. **Slow performance**:
   - First run may be slow due to model downloads
   - Consider using a GPU for faster embeddings

4. **Memory issues**:
   - Try processing smaller documents
   - Reduce the chunk size in settings

## 🔬 Technical Approach

Our legal document analyzer uses a multi-layered approach combining traditional NLP with modern AI:

### 1. **Dual Analysis Engine**
- **Pattern-Based Analysis**: Uses regex patterns and NLP to detect known risky clause types
- **AI-Enhanced Analysis**: Leverages GPT-4 for contextual understanding and nuanced risk assessment

### 2. **Vector Database Integration**
- **ChromaDB**: Stores document embeddings for similarity search
- **Sentence Transformers**: Creates semantic embeddings of legal clauses
- **Context Enhancement**: Uses similar clauses from database to improve analysis accuracy

### 3. **Risk Assessment Framework**
- **Multi-Level Scoring**: High/Medium/Low risk classification
- **Contextual Recommendations**: Provides specific negotiation strategies
- **Alternative Clauses**: Suggests safer alternatives for risky terms

### 4. **Modern UI/UX**
- **3D Glassmorphism Design**: Modern, professional interface
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Real-time Processing**: Streamlit-based responsive web application

## 📈 Future Enhancements

- [ ] Multi-language support for international contracts
- [ ] Export analysis reports to PDF with detailed recommendations
- [ ] Batch processing for multiple documents
- [ ] Custom clause templates and risk patterns
- [ ] Integration with legal databases and case law
- [ ] Mobile app version with offline capabilities
- [ ] Contract comparison and diff analysis
- [ ] Legal precedent search and citation

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is for educational and personal use. Please ensure you have the right to analyze any documents you upload.

## 🔒 Security & Privacy

- **Local Processing**: All document analysis happens on your machine
- **No Data Transmission**: Documents are not sent to external servers (except OpenAI API for AI analysis)
- **API Key Security**: Store your OpenAI API key securely in `.env` file
- **Temporary Storage**: Uploaded files are processed and immediately deleted
- **Vector Database**: Stored locally in `./legal_vector_db`

## 🙏 Acknowledgments

- **UI Framework**: [Streamlit](https://streamlit.io/) for rapid web app development
- **AI Analysis**: [OpenAI GPT-4](https://openai.com/) for advanced legal reasoning
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) for semantic similarity
- **PDF Processing**: [PyMuPDF4LLM](https://pymupdf.readthedocs.io/) for text extraction
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) for similarity search
- **Visualizations**: [Plotly](https://plotly.com/) for interactive 3D charts
- **NLP**: [spaCy](https://spacy.io/) and [NLTK](https://www.nltk.org/) for text processing

---

**Made with ❤️ for demystifying legal documents and empowering better contract negotiations**
