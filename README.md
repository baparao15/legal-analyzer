# Legal Document Analyzer 📜⚖️

An AI-powered legal document analyzer with a beautiful 3D UI that helps identify risky clauses in legal agreements and provides actionable recommendations.

![Legal Document Analyzer](https://img.shields.io/badge/AI-Powered-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

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
cd C:\Users\pendy\Desktop\hello\ml\nextwave\pro
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model** (optional but recommended):
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

1. **Run the application**:
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a PDF** legal document using the sidebar

4. **Click "Analyze Document"** to start the analysis

5. **Explore the results** in the different tabs:
   - 📋 **Analysis**: Overview and risk summary
   - 🔍 **Risk Details**: Detailed breakdown of each risky clause
   - 📊 **Visualizations**: 3D charts and graphs
   - 💾 **History**: Previously analyzed documents

## 📁 Project Structure

```
pro/
├── app.py                    # Main Streamlit application
├── legal_risk_analyzer.py    # Risk detection engine
├── pdf_vector_pipeline.py    # PDF processing and vector storage
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── legal/                   # Folder containing legal documents
    ├── document1.pdf
    ├── document2.pdf
    └── ...
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

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Export analysis reports to PDF
- [ ] Batch processing for multiple documents
- [ ] Custom clause templates
- [ ] Integration with legal databases
- [ ] Mobile app version

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is for educational and personal use. Please ensure you have the right to analyze any documents you upload.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Sentence Transformers](https://www.sbert.net/)
- PDF processing by [PyMuPDF](https://pymupdf.readthedocs.io/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)

---

Made with ❤️ for demystifying legal documents
