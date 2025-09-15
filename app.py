import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import time
from datetime import datetime
import warnings

# Suppress PyTorch warnings that cause Streamlit issues
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Handle PyTorch import gracefully
try:
    import torch
    # Set torch to use CPU only to avoid GPU-related issues
    torch.set_num_threads(1)
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    st.warning(f"PyTorch not available: {e}")

# Import our custom modules
from pdf_vector_pipeline import PDFVectorPipeline, ChunkConfig
from legal_risk_analyzer import LegalRiskAnalyzer, RiskClause
from openai_legal_analyzer import OpenAILegalAnalyzer, AIRiskClause
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for 3D effects and modern design
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Better button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3) !important;
        transform: translateY(0) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(45deg, #00c851, #00ff88);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 200, 81, 0.4);
    }
    
    .status-warning {
        background: linear-gradient(45deg, #f7971e, #ffd200);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 210, 0, 0.4);
    }
    
    .status-error {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }
    
    /* Enhanced file uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* 3D Card Effect */
    .card-3d {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 0 0 1px rgba(255, 255, 255, 0.1);
        transform: perspective(1000px) rotateX(0deg);
        transition: all 0.3s ease;
    }
    
    .card-3d:hover {
        transform: perspective(1000px) rotateX(-5deg) translateY(-10px);
        box-shadow: 
            0 20px 40px 0 rgba(31, 38, 135, 0.5),
            inset 0 0 0 1px rgba(255, 255, 255, 0.2);
    }
    
    /* Risk Level Badges with 3D effect */
    .risk-high {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        transform: translateZ(20px);
    }
    
    .risk-medium {
        background: linear-gradient(45deg, #f7971e, #ffd200);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 210, 0, 0.4);
        transform: translateZ(20px);
    }
    
    .risk-low {
        background: linear-gradient(45deg, #00c851, #00ff88);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 200, 81, 0.4);
        transform: translateZ(20px);
    }
    
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* File Uploader 3D Style */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    /* Progress Bar 3D */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar 3D Effect */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers with 3D text */
    h1, h2, h3 {
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #fff, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.3);
        color: white;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Spinner animation */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'analyzed_documents' not in st.session_state:
        st.session_state.analyzed_documents = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'vector_pipeline' not in st.session_state:
        st.session_state.vector_pipeline = None
    if 'risk_analyzer' not in st.session_state:
        st.session_state.risk_analyzer = LegalRiskAnalyzer()
    if 'ai_analyzer' not in st.session_state:
        try:
            st.session_state.ai_analyzer = OpenAILegalAnalyzer()
            st.session_state.ai_available = True
        except Exception as e:
            st.session_state.ai_analyzer = None
            st.session_state.ai_available = False
            st.warning(f"OpenAI integration not available: {str(e)}")

# Create 3D risk visualization
def create_3d_risk_visualization(risk_score):
    """Create a 3D gauge chart for risk visualization"""
    
    # Create figure
    fig = go.Figure()
    
    # Add 3D scatter plot for visual effect
    theta = [0, 45, 90, 135, 180, 225, 270, 315]
    r = [risk_score['overall_score']] * 8
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        fill='toself',
        name='Risk Score',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgba(102, 126, 234, 0.8)', width=3)
    ))
    
    # Update layout for 3D effect
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False,
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color='rgba(255, 255, 255, 0.7)')
            ),
            angularaxis=dict(
                showline=False,
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color='rgba(255, 255, 255, 0.7)')
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Create risk distribution chart
def create_risk_distribution_chart(risky_clauses):
    """Create a 3D pie chart for risk distribution"""
    
    risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for clause in risky_clauses:
        risk_counts[clause.risk_level.capitalize()] += 1
    
    # Create 3D pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        hole=0.4,
        marker=dict(
            colors=['#ff416c', '#ffd200', '#00c851'],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=2)
        ),
        textfont=dict(size=16, color='white'),
        hoverinfo='label+percent',
        textinfo='value+percent'
    )])
    
    fig.update_layout(
        showlegend=True,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            font=dict(color='rgba(255, 255, 255, 0.8)'),
            bgcolor='rgba(0, 0, 0, 0)'
        )
    )
    
    return fig

# Display risky clause card
def display_risk_clause_card(clause: RiskClause, index: int):
    """Display a single risk clause with improved formatting and alternatives"""
    
    # Risk level styling
    risk_colors = {
        'high': {'bg': '#ff4757', 'icon': '🔴'},
        'medium': {'bg': '#ffa502', 'icon': '🟡'}, 
        'low': {'bg': '#2ed573', 'icon': '🟢'}
    }
    
    risk_info = risk_colors.get(clause.risk_level, risk_colors['medium'])
    
    # Clean and escape text for HTML
    clean_text = clause.text.replace('"', '&quot;').replace('\n', ' ').strip()
    clean_explanation = clause.explanation.replace('"', '&quot;').replace('\n', ' ')
    
    # Create expandable card
    with st.expander(f"{risk_info['icon']} {index}. {clause.clause_type} - {clause.risk_level.upper()} RISK", expanded=True):
        
        # Risk explanation
        st.markdown(f"""
        <div style="background: rgba(255, 165, 2, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #ffa502;">
            <strong style="color: #ffa502;">⚠️ Risk Explanation:</strong><br>
            <span style="color: rgba(255, 255, 255, 0.9);">{clean_explanation}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Problematic clause text
        st.markdown(f"""
        <div style="background: rgba(255, 71, 87, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #ff4757;">
            <strong style="color: #ff4757;">❌ Problematic Statement:</strong><br>
            <em style="color: rgba(255, 255, 255, 0.8);">"{clean_text[:300]}{'...' if len(clean_text) > 300 else ''}"</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative suggestions
        alternatives = get_alternative_clauses(clause.clause_type.lower().replace(" ", "_"))
        if alternatives:
            st.markdown("**✅ Suggested Alternative Statements:**")
            for i, alt in enumerate(alternatives, 1):
                st.markdown(f"""
                <div style="background: rgba(46, 213, 115, 0.1); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #2ed573;">
                    <strong style="color: #2ed573;">Option {i}:</strong><br>
                    <span style="color: rgba(255, 255, 255, 0.9);">"{alt}"</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("**💡 Negotiation Recommendations:**")
        for rec in clause.recommendations:
            st.markdown(f"• {rec}")

def get_alternative_clauses(clause_type: str) -> list:
    """Get alternative clause suggestions for different risk types"""
    alternatives = {
        "arbitration": [
            "Any dispute shall be resolved through mutual agreement, and if unsuccessful, through mediation followed by arbitration with mutually agreed rules and location.",
            "Disputes may be resolved through arbitration, provided both parties agree to the arbitrator and location, with each party bearing their own costs.",
            "Either party may choose between arbitration or court proceedings for dispute resolution, with arbitration limited to disputes under $50,000."
        ],
        "unlimited_liability": [
            "Each party's liability shall be limited to the total amount paid under this agreement in the twelve months preceding the claim.",
            "Liability for damages shall be capped at $[amount], except for cases of gross negligence or willful misconduct.",
            "Both parties agree to mutual limitation of liability, excluding only breaches of confidentiality and intellectual property violations."
        ],
        "unilateral_termination": [
            "Either party may terminate this agreement with 30 days written notice, with obligations continuing until the notice period expires.",
            "Termination requires 60 days notice and completion of ongoing projects, with fair compensation for work performed.",
            "This agreement may be terminated by mutual consent or by either party for material breach after a 30-day cure period."
        ],
        "auto_renewal": [
            "This agreement shall expire on [date] unless both parties agree in writing to extend the term.",
            "The agreement may be renewed for additional one-year terms upon mutual written agreement at least 60 days before expiration.",
            "Either party may provide 90 days notice of non-renewal, after which the agreement will terminate naturally."
        ],
        "ip_assignment": [
            "Each party retains ownership of their pre-existing intellectual property and grants the other party a limited license for project purposes.",
            "Work product shall be jointly owned, with each party having the right to use and license the results independently.",
            "Contractor grants Client a perpetual, non-exclusive license to use deliverables, while retaining ownership and the right to create similar works."
        ]
    }
    return alternatives.get(clause_type, [])

# Main application
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Header with 3D effect
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
            ⚖️ Legal Document Analyzer
        </h1>
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.2rem;">
            AI-Powered Risk Detection for Legal Agreements
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a legal document for analysis"
        )
        
        st.markdown("---")
        
        st.markdown("### 📚 Legal Database")
        
        if st.button("🔄 Process Legal Folder", help="Process all PDFs in legal folder into vector database", use_container_width=True):
            with st.spinner("Processing legal documents..."):
                legal_folder = Path("legal")
                if legal_folder.exists():
                    # Initialize vector pipeline if not already done
                    if st.session_state.vector_pipeline is None:
                        try:
                            st.session_state.vector_pipeline = PDFVectorPipeline(
                                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                                vector_db_path="./legal_vector_db",
                                collection_name="legal_docs"
                            )
                        except Exception as e:
                            st.error(f"Failed to initialize vector pipeline: {e}")
                            st.stop()
                    
                    processed_count = 0
                    total_files = len(list(legal_folder.glob("*.pdf")))
                    
                    if total_files == 0:
                        st.warning("No PDF files found in legal folder")
                        st.stop()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, pdf_file in enumerate(legal_folder.glob("*.pdf")):
                        status_text.text(f"Processing {pdf_file.name}...")
                        try:
                            if st.session_state.vector_pipeline and st.session_state.vector_pipeline.embeddings_available:
                                result = st.session_state.vector_pipeline.process_pdf(str(pdf_file))
                                if result["status"] == "success":
                                    processed_count += 1
                                else:
                                    st.warning(f"Failed to process {pdf_file.name}: {result.get('message', 'Unknown error')}")
                            else:
                                st.warning("Vector pipeline not available - embeddings disabled")
                                break
                        except Exception as e:
                            st.error(f"Error processing {pdf_file.name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if processed_count > 0:
                        st.success(f"✅ Processed {processed_count}/{total_files} legal documents")
                        st.rerun()  # Refresh to update the document count
                    else:
                        st.error("No documents were successfully processed")
                else:
                    st.error("Legal folder not found")
        
        # Show vector database stats with better error handling
        try:
            if st.session_state.vector_pipeline and hasattr(st.session_state.vector_pipeline, 'collection'):
                collection_count = st.session_state.vector_pipeline.collection.count()
                st.metric("Documents in Database", collection_count, delta=None)
            else:
                st.metric("Documents in Database", 0, help="Click 'Process Legal Folder' to add documents")
        except Exception as e:
            st.metric("Documents in Database", "Error", help=f"Database error: {str(e)}")
        
        # Add refresh button for database stats
        if st.button("🔄 Refresh Stats", help="Refresh database statistics", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Analysis Settings")
        
        use_openai_analysis = st.checkbox(
            "Use AI Enhanced Analysis",
            value=st.session_state.get('ai_available', False),
            disabled=not st.session_state.get('ai_available', False),
            help="Use GPT-4 for advanced legal risk analysis"
        )
        
        enable_vector_search = st.checkbox(
            "Use Vector Database Context",
            value=True,
            help="Use similar clauses from legal database for enhanced analysis"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Statistics")
        if st.session_state.analyzed_documents:
            st.metric("Documents Analyzed", len(st.session_state.analyzed_documents))
            
            total_risks = sum(doc['total_risks'] for doc in st.session_state.analyzed_documents)
            st.metric("Total Risks Found", total_risks)
    
    # Main content area
    if uploaded_file is not None:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📋 Analysis", "🔍 Risk Details", "📄 Document View"])
        
        with tab1:
            # Simplified Analysis Tab
            if st.session_state.current_analysis:
                analysis = st.session_state.current_analysis
                
                # Document Header
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
                           border-radius: 15px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.2);">
                    <h3 style="color: white; margin: 0;">📄 {uploaded_file.name}</h3>
                    <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">Size: {uploaded_file.size / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Summary Cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = analysis['risk_score']['risk_level'].upper()
                    risk_color = {'HIGH': '#ff4757', 'MEDIUM': '#ffa502', 'LOW': '#2ed573'}.get(risk_level, '#ffa502')
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {risk_color}20, {risk_color}10); 
                               border: 1px solid {risk_color}40; border-radius: 12px; padding: 1rem; text-align: center;">
                        <h2 style="color: {risk_color}; margin: 0; font-size: 2rem;">{analysis['risk_score']['overall_score']}</h2>
                        <p style="color: rgba(255,255,255,0.8); margin: 0;">Risk Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {risk_color}20, {risk_color}10); 
                               border: 1px solid {risk_color}40; border-radius: 12px; padding: 1rem; text-align: center;">
                        <h2 style="color: {risk_color}; margin: 0; font-size: 1.5rem;">{risk_level}</h2>
                        <p style="color: rgba(255,255,255,0.8); margin: 0;">Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_risks = analysis['risk_score']['total_risks']
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                               border: 1px solid rgba(255,255,255,0.2); border-radius: 12px; padding: 1rem; text-align: center;">
                        <h2 style="color: white; margin: 0; font-size: 2rem;">{total_risks}</h2>
                        <p style="color: rgba(255,255,255,0.8); margin: 0;">Total Risks</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk Breakdown
                if total_risks > 0:
                    st.markdown("### 📊 Risk Breakdown")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_count = analysis['risk_score']['high_risk_count']
                        st.metric("🔴 High Risk", high_count)
                    
                    with col2:
                        medium_count = analysis['risk_score']['medium_risk_count']
                        st.metric("🟡 Medium Risk", medium_count)
                    
                    with col3:
                        low_count = analysis['risk_score']['low_risk_count']
                        st.metric("🟢 Low Risk", low_count)
                    
                    # Quick Risk Summary
                    st.markdown("### 🎯 Quick Summary")
                    for clause in analysis['risky_clauses'][:3]:  # Show top 3 risks
                        location_text = ""
                        if hasattr(clause, 'page_location') and hasattr(clause, 'section_location'):
                            location_text = f" (Page ~{clause.page_location}, {clause.section_location})"
                        
                        risk_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(clause.risk_level, '🟡')
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                            <strong>{risk_icon} {clause.clause_type}</strong>{location_text}<br>
                            <small style="color: rgba(255,255,255,0.7);">{clause.explanation}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(analysis['risky_clauses']) > 3:
                        st.info(f"+ {len(analysis['risky_clauses']) - 3} more risks found. Check the 'Risk Details' tab for complete analysis.")
                
                else:
                    st.success("✅ No significant risks detected in this document!")
                
            else:
                # Upload and Analyze Section
                st.markdown("### 📤 Upload Document for Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                               border-radius: 15px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.2);">
                        <h4 style="color: white; margin: 0;">📄 {uploaded_file.name}</h4>
                        <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">Size: {uploaded_file.size / 1024:.1f} KB</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🚀 Analyze Document", use_container_width=True, type="primary"):
                        with st.spinner("🔍 Analyzing document..."):
                            # Save uploaded file temporarily
                            temp_path = Path(f"temp_{uploaded_file.name}")
                            temp_path.write_bytes(uploaded_file.getvalue())
                            
                            try:
                                # Initialize vector pipeline if enabled
                                if enable_vector_search and st.session_state.vector_pipeline is None:
                                    st.session_state.vector_pipeline = PDFVectorPipeline(
                                        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                                        vector_db_path="./legal_vector_db",
                                        collection_name="legal_docs"
                                    )
                                
                                # Extract text
                                progress_bar = st.progress(0)
                                progress_bar.progress(20, text="Extracting text from PDF...")
                                
                                if enable_vector_search:
                                    text = st.session_state.vector_pipeline.extract_text_from_pdf(str(temp_path))
                                else:
                                    import pymupdf4llm
                                    text = pymupdf4llm.to_markdown(str(temp_path))
                                
                                # Analyze for risks
                                progress_bar.progress(60, text="Analyzing for risky clauses...")
                                
                                # Get vector database context if enabled
                                vector_context = None
                                if enable_vector_search and st.session_state.vector_pipeline:
                                    progress_bar.progress(65, text="Searching vector database...")
                                    try:
                                        sample_text = text[:1000]
                                        vector_results = st.session_state.vector_pipeline.search_similar(sample_text, top_k=5)
                                        vector_context = vector_results['results']
                                    except Exception as e:
                                        st.warning(f"Vector search failed: {str(e)}")
                                
                                if use_openai_analysis and st.session_state.ai_available:
                                    # AI analysis
                                    progress_bar.progress(70, text="Running AI analysis...")
                                    
                                    enhanced_text = text
                                    if vector_context:
                                        context_clauses = "\n\n--- Similar Legal Clauses from Database ---\n"
                                        for ctx in vector_context[:3]:
                                            context_clauses += f"\n{ctx['chunk']}\n"
                                        enhanced_text = context_clauses + "\n\n--- Current Document ---\n" + text
                                    
                                    ai_risky_clauses = st.session_state.ai_analyzer.analyze_document_with_ai(enhanced_text, "legal_contract")
                                    
                                    risky_clauses = []
                                    for ai_clause in ai_risky_clauses:
                                        standard_clause = RiskClause(
                                            clause_type=ai_clause.clause_type,
                                            risk_level=ai_clause.risk_level,
                                            text=ai_clause.text,
                                            location=ai_clause.location,
                                            explanation=ai_clause.explanation,
                                            recommendations=ai_clause.recommendations
                                        )
                                        risky_clauses.append(standard_clause)
                                    
                                    progress_bar.progress(85, text="Calculating risk score...")
                                    risk_score = st.session_state.ai_analyzer.calculate_ai_risk_score(ai_risky_clauses)
                                    
                                    progress_bar.progress(90, text="Generating summary...")
                                    ai_summary = st.session_state.ai_analyzer.generate_ai_summary(ai_risky_clauses, risk_score)
                                    
                                    analysis_result = {
                                        'filename': uploaded_file.name,
                                        'timestamp': datetime.now(),
                                        'risky_clauses': risky_clauses,
                                        'risk_score': risk_score,
                                        'total_risks': len(risky_clauses),
                                        'text_preview': text[:500],
                                        'full_text': text,
                                        'ai_enhanced': True,
                                        'ai_summary': ai_summary,
                                        'vector_context': vector_context
                                    }
                                    
                                else:
                                    # Traditional analysis
                                    progress_bar.progress(70, text="Running pattern analysis...")
                                    
                                    analyzer = LegalRiskAnalyzer()
                                    risky_clauses = analyzer.analyze_document(text)
                                    
                                    progress_bar.progress(85, text="Calculating risk score...")
                                    risk_score = analyzer.calculate_risk_score(risky_clauses)
                                    
                                    analysis_result = {
                                        'filename': uploaded_file.name,
                                        'timestamp': datetime.now(),
                                        'risky_clauses': risky_clauses,
                                        'risk_score': risk_score,
                                        'total_risks': len(risky_clauses),
                                        'text_preview': text[:500],
                                        'full_text': text,
                                        'ai_enhanced': False,
                                        'ai_summary': None,
                                        'vector_context': vector_context
                                    }
                                
                                st.session_state.current_analysis = analysis_result
                                st.session_state.analyzed_documents.append(analysis_result)
                                
                                temp_path.unlink()
                                progress_bar.empty()
                                
                                st.success("✅ Analysis complete!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error during analysis: {str(e)}")
                                if temp_path.exists():
                                    temp_path.unlink()
        
        with tab2:
            if st.session_state.current_analysis:
                analysis = st.session_state.current_analysis
                
                # Filter options
                col1, col2 = st.columns([2, 1])
                with col1:
                    search_term = st.text_input("🔎 Search in risks", placeholder="Enter keywords...")
                
                with col2:
                    risk_filter = st.selectbox(
                        "Filter by risk level",
                        ["All", "High", "Medium", "Low"]
                    )
                
                # Display risky clauses
                filtered_clauses = analysis['risky_clauses']
                
                if risk_filter != "All":
                    filtered_clauses = [c for c in filtered_clauses if c.risk_level == risk_filter.lower()]
                
                if search_term:
                    filtered_clauses = [c for c in filtered_clauses if search_term.lower() in c.text.lower() or search_term.lower() in c.explanation.lower()]
                
                if filtered_clauses:
                    for i, clause in enumerate(filtered_clauses, 1):
                        display_risk_clause_card(clause, i)
                else:
                    st.info("No risks found matching your criteria.")
            else:
                st.info("Please analyze a document first.")
        
        with tab3:
            if st.session_state.current_analysis:
                analysis = st.session_state.current_analysis
                
                st.markdown("### 📄 Document with Highlighted Risks")
                
                if analysis.get('highlight_enabled', False) and analysis.get('full_text'):
                    # Create highlighted version of the document
                    highlighted_text = analysis['full_text']
                    
                    # Sort clauses by location to avoid overlapping highlights
                    sorted_clauses = sorted(analysis['risky_clauses'], key=lambda x: x.location, reverse=True)
                    
                    for clause in sorted_clauses:
                        if clause.text in highlighted_text:
                            risk_color = {
                                'high': '#ff416c',
                                'medium': '#ffd200', 
                                'low': '#00c851'
                            }.get(clause.risk_level, '#ffd200')
                            
                            # Create highlighted version
                            highlighted_clause = f'<mark style="background-color: {risk_color}; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;" title="{clause.clause_type}: {clause.explanation[:100]}...">{clause.text[:200]}...</mark>'
                            
                            # Replace in text (only first occurrence to avoid duplicates)
                            highlighted_text = highlighted_text.replace(clause.text[:200], highlighted_clause, 1)
                    
                    # Display the highlighted document
                    st.markdown("#### 📖 Document Text with Risk Highlights")
                    st.markdown(
                        f'<div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 10px; max-height: 600px; overflow-y: auto; line-height: 1.6; color: rgba(255,255,255,0.9);">{highlighted_text}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Legend
                    st.markdown("#### 🎨 Risk Level Legend")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<span style="background-color: #ff416c; color: white; padding: 4px 8px; border-radius: 4px;">🔴 High Risk</span>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<span style="background-color: #ffd200; color: black; padding: 4px 8px; border-radius: 4px;">🟡 Medium Risk</span>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<span style="background-color: #00c851; color: white; padding: 4px 8px; border-radius: 4px;">🟢 Low Risk</span>', unsafe_allow_html=True)
                    
                    # Show vector context if available
                    if analysis.get('vector_context'):
                        st.markdown("---")
                        st.markdown("#### 🔍 Similar Clauses from Legal Database")
                        
                        for i, ctx in enumerate(analysis['vector_context'][:3], 1):
                            with st.expander(f"Similar Clause {i} (Similarity: {1-ctx['distance']:.2f})"):
                                st.markdown(f"**Source:** {ctx['metadata'].get('source_file', 'Unknown')}")
                                st.markdown(f"**Text:** {ctx['chunk']}")
                
                else:
                    st.info("Document highlighting is disabled. Enable 'Highlight Risky Clauses' in the sidebar to see highlighted text.")
                    
                    # Show plain text as fallback
                    if analysis.get('full_text'):
                        with st.expander("📄 View Plain Document Text"):
                            st.text_area("Document Content", analysis['full_text'], height=400)
            else:
                st.info("Please analyze a document first to view highlighted text.")
    
    else:
        # Welcome screen with 3D animation
        st.markdown("""
        <div class="card-3d" style="text-align: center; padding: 4rem;">
            <h2 style="color: white; margin-bottom: 1rem;">Welcome to Legal Document Analyzer</h2>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;">
                Upload a legal document to identify potential risks and get AI-powered recommendations.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔍</div>
                    <p style="color: rgba(255, 255, 255, 0.8);">AI-Powered Analysis</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
                    <p style="color: rgba(255, 255, 255, 0.8);">Real-time Results</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
                    <p style="color: rgba(255, 255, 255, 0.8);">3D Visualizations</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card-3d">
                <h3 style="color: white;">🎯 Risk Detection</h3>
                <p style="color: rgba(255, 255, 255, 0.8);">
                    Identifies potentially risky clauses including unlimited liability, 
                    unfavorable termination terms, and IP assignments.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card-3d">
                <h3 style="color: white;">💡 Smart Recommendations</h3>
                <p style="color: rgba(255, 255, 255, 0.8);">
                    Get actionable recommendations for each identified risk to protect 
                    your interests in negotiations.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card-3d">
                <h3 style="color: white;">📈 Visual Analytics</h3>
                <p style="color: rgba(255, 255, 255, 0.8);">
                    Beautiful 3D visualizations help you understand the risk profile 
                    of your documents at a glance.
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
