@echo off
echo ======================================
echo Legal Document Analyzer 
echo ======================================
echo.

REM Use existing Anaconda environment
echo Using Anaconda environment...

REM Check if requirements are installed
echo Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Check for spaCy model (optional - app will work without it)
echo Checking spaCy language model...
python -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
if errorlevel 1 (
    echo spaCy model not found - app will use NLTK fallback
    echo You can manually install it later with: python -m spacy download en_core_web_sm
    echo.
)

REM Check OpenAI API configuration
echo Checking OpenAI configuration...
if exist ".env" (
    echo ✅ Environment file found
) else (
    echo ⚠️ .env file not found - OpenAI features may not work
)

REM Run the application
echo Starting Legal Document Analyzer...
echo.
echo 🚀 The application will open in your default browser
echo 🛑 Press Ctrl+C to stop the application
echo.
streamlit run app.py --server.headless false

pause
