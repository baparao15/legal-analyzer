import os
import openai
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import logging
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIRiskClause:
    """Enhanced risk clause with AI analysis"""
    clause_type: str
    risk_level: str  # "high", "medium", "low"
    text: str
    location: int
    explanation: str
    recommendations: List[str]
    confidence_score: float
    legal_precedents: List[str]
    severity_reasoning: str

class OpenAILegalAnalyzer:
    """Enhanced legal document analyzer using OpenAI GPT"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI legal analyzer"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.openai_available = False
        
        if self.api_key:
            try:
                # Initialize OpenAI client
                openai.api_key = self.api_key
                self.client = openai.OpenAI(api_key=self.api_key)
                
                # Test API availability
                test_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                self.openai_available = True
                logger.info("OpenAI Legal Analyzer initialized successfully")
            except openai.AuthenticationError:
                logger.warning("OpenAI API key authentication failed")
                self.openai_available = False
            except Exception as e:
                logger.warning(f"OpenAI API not available: {e}")
                self.openai_available = False
        else:
            logger.warning("No OpenAI API key provided")
            self.openai_available = False
    
    def analyze_document_with_ai(self, text: str, document_type: str = "general") -> List[AIRiskClause]:
        """Analyze document using OpenAI GPT for enhanced risk detection"""
        logger.info("Starting AI-powered legal document analysis...")
        
        if not self.openai_available:
            logger.warning("OpenAI not available, using fallback pattern-based analysis")
            return self._fallback_pattern_analysis(text)
        
        # Split document into chunks for analysis
        chunks = self._split_document(text)
        all_risks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
            chunk_risks = self._analyze_chunk(chunk, document_type, i * 1000)
            all_risks.extend(chunk_risks)
        
        # Deduplicate and rank risks
        final_risks = self._deduplicate_and_rank_risks(all_risks)
        
        logger.info(f"AI analysis complete. Found {len(final_risks)} risk clauses")
        return final_risks
    
    def _split_document(self, text: str, chunk_size: int = 3000) -> List[str]:
        """Split document into manageable chunks for AI analysis"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _make_api_call_with_retry(self, messages: List[Dict], max_tokens: int = 2000, max_retries: int = 3):
        """Make OpenAI API call with retry logic for rate limits"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                return response
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 1 + random.uniform(0, 1)
                    logger.warning(f"API error: {e}, retrying in {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                else:
                    raise e

    def _analyze_chunk(self, chunk: str, document_type: str, offset: int) -> List[AIRiskClause]:
        """Analyze a single chunk using OpenAI"""
        prompt = f"""
        You are an expert legal analyst. Analyze the following legal document excerpt for potentially risky clauses.
        Document type: {document_type}
        
        For each risky clause you identify, provide:
        1. Clause type (e.g., "Unlimited Liability", "Unilateral Termination")
        2. Risk level (high/medium/low)
        3. Exact text of the risky clause
        4. Detailed explanation of why it's risky
        5. Specific recommendations to mitigate the risk
        6. Confidence score (0.0-1.0)
        7. Relevant legal precedents or principles
        8. Reasoning for severity assessment
        
        Focus on these high-risk areas:
        - Liability and indemnification clauses
        - Termination and renewal terms
        - Intellectual property assignments
        - Non-compete and confidentiality agreements
        - Payment and refund terms
        - Warranty disclaimers
        - Dispute resolution mechanisms
        - Force majeure clauses
        
        Document excerpt:
        {chunk}
        
        Respond in JSON format with an array of risk objects:
        {{
            "risks": [
                {{
                    "clause_type": "string",
                    "risk_level": "high|medium|low",
                    "text": "exact clause text",
                    "explanation": "detailed explanation",
                    "recommendations": ["recommendation1", "recommendation2"],
                    "confidence_score": 0.0-1.0,
                    "legal_precedents": ["precedent1", "precedent2"],
                    "severity_reasoning": "why this severity level"
                }}
            ]
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert legal analyst specializing in contract risk assessment."},
                {"role": "user", "content": prompt}
            ]
            response = self._make_api_call_with_retry(messages, max_tokens=2000)
            
            # Parse the JSON response
            content = response.choices[0].message.content
            
            # Clean up the response to ensure valid JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content.strip())
            
            # Convert to AIRiskClause objects
            risk_clauses = []
            for risk_data in result.get("risks", []):
                try:
                    risk_clause = AIRiskClause(
                        clause_type=risk_data.get("clause_type", "Unknown"),
                        risk_level=risk_data.get("risk_level", "medium"),
                        text=risk_data.get("text", ""),
                        location=offset + chunk.find(risk_data.get("text", "")[:50]),
                        explanation=risk_data.get("explanation", ""),
                        recommendations=risk_data.get("recommendations", []),
                        confidence_score=float(risk_data.get("confidence_score", 0.5)),
                        legal_precedents=risk_data.get("legal_precedents", []),
                        severity_reasoning=risk_data.get("severity_reasoning", "")
                    )
                    risk_clauses.append(risk_clause)
                except Exception as e:
                    logger.warning(f"Error parsing risk clause: {e}")
                    continue
            
            return risk_clauses
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return []
    
    def _deduplicate_and_rank_risks(self, risks: List[AIRiskClause]) -> List[AIRiskClause]:
        """Remove duplicates and rank risks by severity and confidence"""
        if not risks:
            return []
        
        # Remove duplicates based on similar text content
        unique_risks = []
        seen_texts = set()
        
        for risk in risks:
            # Create a simplified version of the text for comparison
            simplified_text = ' '.join(risk.text.lower().split()[:10])
            
            if simplified_text not in seen_texts:
                seen_texts.add(simplified_text)
                unique_risks.append(risk)
        
        # Sort by risk level and confidence score
        risk_priority = {"high": 3, "medium": 2, "low": 1}
        unique_risks.sort(
            key=lambda x: (risk_priority.get(x.risk_level, 1), x.confidence_score),
            reverse=True
        )
        
        return unique_risks
    
    def _fallback_pattern_analysis(self, text: str) -> List[AIRiskClause]:
        """Fallback pattern-based analysis when OpenAI is not available"""
        from legal_risk_analyzer import LegalRiskAnalyzer
        
        # Use the traditional analyzer as fallback
        traditional_analyzer = LegalRiskAnalyzer()
        traditional_risks = traditional_analyzer.analyze_document(text)
        
        # Convert to AIRiskClause format
        ai_risks = []
        for risk in traditional_risks:
            ai_risk = AIRiskClause(
                clause_type=risk.clause_type,
                risk_level=risk.risk_level,
                text=risk.text,
                location=risk.location,
                explanation=risk.explanation,
                recommendations=risk.recommendations,
                confidence_score=0.8,  # Default confidence for pattern matching
                legal_precedents=["Pattern-based detection"],
                severity_reasoning=f"Detected through pattern matching: {risk.clause_type}"
            )
            ai_risks.append(ai_risk)
        
        return ai_risks
    
    def generate_ai_summary(self, risks: List[AIRiskClause], document_name: str) -> str:
        """Generate a comprehensive AI-powered legal analysis summary (200-300 lines)"""
        if not risks:
            return "No significant risks identified in the document."
        
        # Prepare detailed risk data for comprehensive summary
        risk_summary = {
            "high_risks": [r for r in risks if r.risk_level == "high"],
            "medium_risks": [r for r in risks if r.risk_level == "medium"],
            "low_risks": [r for r in risks if r.risk_level == "low"]
        }
        
        # Create detailed risk descriptions
        high_risk_details = []
        for i, risk in enumerate(risk_summary['high_risks'][:5], 1):
            high_risk_details.append(f"""
            {i}. {risk.clause_type}
               - Risk Level: {risk.risk_level.upper()}
               - Confidence: {risk.confidence_score:.2f}
               - Explanation: {risk.explanation}
               - Recommendations: {'; '.join(risk.recommendations[:3])}
               - Legal Precedents: {'; '.join(risk.legal_precedents[:2]) if risk.legal_precedents else 'None specified'}
            """)
        
        medium_risk_details = []
        for i, risk in enumerate(risk_summary['medium_risks'][:3], 1):
            medium_risk_details.append(f"""
            {i}. {risk.clause_type}
               - Explanation: {risk.explanation}
               - Key Recommendations: {'; '.join(risk.recommendations[:2])}
            """)
        
        prompt = f"""
        Generate a comprehensive legal risk analysis report for "{document_name}". 
        This should be a detailed, professional report of approximately 200-300 lines covering all aspects of the legal risks identified.
        
        DOCUMENT ANALYSIS OVERVIEW:
        - Total Risks Identified: {len(risks)}
        - High Risk Clauses: {len(risk_summary['high_risks'])}
        - Medium Risk Clauses: {len(risk_summary['medium_risks'])}
        - Low Risk Clauses: {len(risk_summary['low_risks'])}
        
        HIGH-RISK ISSUES DETAILS:
        {''.join(high_risk_details)}
        
        MEDIUM-RISK ISSUES DETAILS:
        {''.join(medium_risk_details)}
        
        Please provide a comprehensive report with the following structure:
        
        1. EXECUTIVE SUMMARY (3-4 paragraphs)
           - Overall risk assessment and key findings
           - Business impact analysis
           - Immediate concerns requiring attention
        
        2. DETAILED RISK ANALYSIS (10-15 paragraphs)
           - Comprehensive analysis of each high-risk clause
           - Legal implications and potential consequences
           - Industry context and regulatory considerations
           - Comparative analysis with standard practices
        
        3. FINANCIAL AND OPERATIONAL IMPACT (5-7 paragraphs)
           - Potential financial exposure
           - Operational constraints and limitations
           - Timeline and milestone risks
           - Performance and delivery obligations
        
        4. LEGAL COMPLIANCE AND REGULATORY CONCERNS (4-5 paragraphs)
           - Regulatory compliance issues
           - Jurisdictional considerations
           - Dispute resolution mechanisms
           - Intellectual property implications
        
        5. STRATEGIC RECOMMENDATIONS (8-10 paragraphs)
           - Priority actions for immediate implementation
           - Negotiation strategies and key talking points
           - Alternative clause suggestions
           - Risk mitigation strategies
           - Long-term contract management recommendations
        
        6. IMPLEMENTATION ROADMAP (3-4 paragraphs)
           - Step-by-step action plan
           - Timeline for addressing critical issues
           - Stakeholder involvement requirements
           - Success metrics and monitoring
        
        7. CONCLUSION AND NEXT STEPS (2-3 paragraphs)
           - Summary of critical actions
           - Decision points for leadership
           - Follow-up requirements
        
        Make this report detailed, professional, and actionable for business stakeholders and legal teams. 
        Use specific examples from the identified risks and provide concrete recommendations.
        Ensure the report is approximately 200-300 lines long with comprehensive coverage of all risk aspects.
        """
        
        try:
            if not self.openai_available:
                return self._generate_fallback_summary(risks, document_name)
            
            messages = [
                {"role": "system", "content": "You are a senior legal advisor and risk analyst providing comprehensive legal analysis reports. Generate detailed, professional reports with specific recommendations and thorough analysis."},
                {"role": "user", "content": prompt}
            ]
            response = self._make_api_call_with_retry(messages, max_tokens=4000)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating comprehensive AI summary: {e}")
            return self._generate_fallback_summary(risks, document_name)
    
    def _generate_fallback_summary(self, risks: List[AIRiskClause], document_name: str) -> str:
        """Generate comprehensive summary without OpenAI"""
        if not risks:
            return "No significant risks identified in the document."
        
        risk_summary = {
            "high_risks": [r for r in risks if r.risk_level == "high"],
            "medium_risks": [r for r in risks if r.risk_level == "medium"],
            "low_risks": [r for r in risks if r.risk_level == "low"]
        }
        
        summary = f"""
COMPREHENSIVE LEGAL RISK ANALYSIS REPORT
Document: {document_name}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Generated by: Pattern-Based Legal Risk Analyzer

========================================
EXECUTIVE SUMMARY
========================================

This comprehensive analysis has identified {len(risks)} potential risk clauses in the document "{document_name}".
The risk distribution includes {len(risk_summary['high_risks'])} high-risk, {len(risk_summary['medium_risks'])} medium-risk, 
and {len(risk_summary['low_risks'])} low-risk clauses.

Overall Risk Assessment: {'HIGH' if len(risk_summary['high_risks']) > 3 else 'MEDIUM' if len(risk_summary['high_risks']) > 0 else 'LOW'}

The document requires immediate attention due to the presence of potentially problematic clauses that could 
expose the organization to significant legal and financial risks. Key areas of concern include liability 
provisions, termination clauses, and indemnification requirements.

Business Impact: The identified risks could result in operational constraints, financial exposure, and 
potential legal disputes if not properly addressed during contract negotiation and execution.

========================================
DETAILED RISK ANALYSIS
========================================
"""
        
        # Add detailed analysis for each high-risk clause
        for i, risk in enumerate(risk_summary['high_risks'][:5], 1):
            summary += f"""
{i}. HIGH RISK: {risk.clause_type}
   Risk Level: {risk.risk_level.upper()}
   Confidence Score: {risk.confidence_score:.2f}
   
   Legal Implications:
   {risk.explanation}
   
   Clause Text:
   "{risk.text[:300]}{'...' if len(risk.text) > 300 else ''}"
   
   Recommendations:
   {chr(10).join(f'   • {rec}' for rec in risk.recommendations)}
   
   Legal Precedents:
   {chr(10).join(f'   • {prec}' for prec in risk.legal_precedents)}
   
   Severity Reasoning:
   {risk.severity_reasoning}
   
"""
        
        # Add medium risks
        if risk_summary['medium_risks']:
            summary += """
========================================
MEDIUM RISK CLAUSES
========================================
"""
            for i, risk in enumerate(risk_summary['medium_risks'][:3], 1):
                summary += f"""
{i}. {risk.clause_type}
   Explanation: {risk.explanation}
   Key Recommendations: {'; '.join(risk.recommendations[:2])}
   
"""
        
        # Financial and operational impact
        summary += f"""
========================================
FINANCIAL AND OPERATIONAL IMPACT
========================================

Potential Financial Exposure:
The identified high-risk clauses could result in significant financial liability, particularly in areas of 
indemnification and unlimited liability provisions. Organizations should budget for potential legal costs 
and consider insurance coverage for identified risks.

Operational Constraints:
Several clauses may impose operational limitations that could affect business flexibility and growth. 
Termination clauses and performance obligations require careful management to avoid breach scenarios.

Timeline and Milestone Risks:
The document contains time-sensitive obligations that require immediate attention and ongoing monitoring. 
Failure to meet specified deadlines could trigger penalty clauses or contract termination.

Performance and Delivery Obligations:
Service level agreements and delivery requirements present operational challenges that need resource 
allocation and performance monitoring systems.

Risk Mitigation Costs:
Implementation of recommended risk mitigation strategies will require investment in legal review, 
compliance systems, and operational procedures.

========================================
LEGAL COMPLIANCE AND REGULATORY CONCERNS
========================================

Regulatory Compliance Issues:
The document should be reviewed for compliance with applicable industry regulations and standards. 
Certain clauses may conflict with regulatory requirements in your jurisdiction.

Jurisdictional Considerations:
Dispute resolution and governing law clauses require careful evaluation to ensure favorable legal positioning. 
Consider the implications of foreign jurisdiction requirements.

Dispute Resolution Mechanisms:
The specified dispute resolution procedures may not be optimal for your organization's interests. 
Alternative dispute resolution methods should be considered.

Intellectual Property Implications:
IP assignment and licensing clauses require thorough review to protect proprietary assets and avoid 
unintended transfers of valuable intellectual property rights.

========================================
STRATEGIC RECOMMENDATIONS
========================================

Priority Actions for Immediate Implementation:
1. Engage qualified legal counsel for detailed review of high-risk clauses
2. Negotiate modifications to unlimited liability provisions
3. Clarify termination procedures and notice requirements
4. Review and modify indemnification clauses
5. Establish performance monitoring systems

Negotiation Strategies and Key Talking Points:
• Request mutual liability caps and carve-outs for certain damages
• Propose alternative dispute resolution mechanisms
• Negotiate more favorable termination notice periods
• Seek reciprocal indemnification provisions
• Include force majeure protections

Alternative Clause Suggestions:
Consider industry-standard alternatives for problematic provisions, including standardized liability 
limitations, balanced termination rights, and mutual indemnification arrangements.

Risk Mitigation Strategies:
Implement comprehensive contract management procedures, establish regular compliance monitoring, 
and maintain adequate insurance coverage for identified risks.

Long-term Contract Management Recommendations:
Develop standardized contract templates, establish regular review cycles, and implement automated 
monitoring systems for key performance indicators and compliance requirements.

========================================
IMPLEMENTATION ROADMAP
========================================

Step-by-Step Action Plan:
Phase 1 (Immediate - 1-2 weeks):
• Legal review of high-risk clauses
• Initial risk assessment and prioritization
• Stakeholder notification and briefing

Phase 2 (Short-term - 2-4 weeks):
• Contract negotiation and clause modification
• Implementation of immediate risk controls
• Documentation of agreed changes

Phase 3 (Medium-term - 1-3 months):
• Full contract execution with modifications
• Implementation of monitoring systems
• Staff training on new procedures

Timeline for Addressing Critical Issues:
High-risk items require resolution within 2 weeks, medium-risk items within 4 weeks, and low-risk 
items should be addressed during the next contract review cycle.

Stakeholder Involvement Requirements:
Legal team, business stakeholders, risk management, and senior leadership should be involved in 
the review and approval process for significant contract modifications.

Success Metrics and Monitoring:
Establish KPIs for contract compliance, risk incident tracking, and performance against agreed terms. 
Regular quarterly reviews should assess the effectiveness of implemented risk controls.

========================================
CONCLUSION AND NEXT STEPS
========================================

Summary of Critical Actions:
This analysis has identified {len(risk_summary['high_risks'])} critical risk areas requiring immediate attention. 
The organization should prioritize legal review and negotiation of high-risk clauses before contract execution.

Decision Points for Leadership:
Senior management must decide on acceptable risk levels, budget allocation for legal review, and 
timeline for contract negotiation. Consider whether the business benefits justify the identified risks.

Follow-up Requirements:
Schedule follow-up legal review after initial modifications, establish ongoing monitoring procedures, 
and plan for regular contract performance assessments. Document all decisions and rationale for 
future reference and audit purposes.

========================================
RISK SUMMARY TABLE
========================================

Total Risks Identified: {len(risks)}
High Risk Clauses: {len(risk_summary['high_risks'])}
Medium Risk Clauses: {len(risk_summary['medium_risks'])}
Low Risk Clauses: {len(risk_summary['low_risks'])}

Recommended Action: {'IMMEDIATE LEGAL REVIEW REQUIRED' if len(risk_summary['high_risks']) > 0 else 'STANDARD REVIEW PROCESS'}

This analysis provides a comprehensive foundation for informed decision-making regarding the legal 
risks associated with this document. All recommendations should be validated with qualified legal 
counsel familiar with your specific business context and applicable law.
"""
        
        return summary
    
    def calculate_ai_risk_score(self, risks: List[AIRiskClause]) -> Dict[str, any]:
        """Calculate enhanced risk score using AI confidence scores"""
        if not risks:
            return {
                "overall_score": 0,
                "risk_level": "low",
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "confidence_weighted_score": 0,
                "total_risks": 0
            }
        
        risk_weights = {"high": 3, "medium": 2, "low": 1}
        
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        total_score = 0
        confidence_weighted_score = 0
        
        for risk in risks:
            risk_counts[risk.risk_level] += 1
            weight = risk_weights[risk.risk_level]
            total_score += weight
            confidence_weighted_score += weight * risk.confidence_score
        
        # Calculate scores
        max_possible_score = len(risks) * 3
        normalized_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        confidence_score = (confidence_weighted_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        # Determine overall risk level
        if normalized_score >= 70:
            overall_risk = "high"
        elif normalized_score >= 40:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_score": round(normalized_score, 2),
            "risk_level": overall_risk,
            "high_risk_count": risk_counts["high"],
            "medium_risk_count": risk_counts["medium"],
            "low_risk_count": risk_counts["low"],
            "confidence_weighted_score": round(confidence_score, 2),
            "total_risks": len(risks)
        }
