import re
import nltk
import spacy
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
try:
    nltk.data.find('averaged_perceptron_tagger')
except:
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskClause:
    """Represents a risky clause found in the document"""
    clause_type: str
    risk_level: str  # "high", "medium", "low"
    text: str
    location: int
    explanation: str
    recommendations: List[str]

class LegalRiskAnalyzer:
    """Advanced legal document analyzer with comprehensive risk assessment capabilities"""
    
    def __init__(self):
        # Initialize NLP with legal-specific models
        try:
            self.nlp = spacy.load("en_legal_ner_sm")
            self.use_spacy = True
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.use_spacy = True
            except:
                self.use_spacy = False
                logger.warning("SpaCy not available, using basic NLTK. Install 'en_legal_ner_sm' for better legal analysis.")
        
        # Legal terminology and patterns
        self.legal_terms = {
            'indemnification': {
                'patterns': [
                    r'indemnif(y|ies|ied|ication)',
                    r'hold\s+harmless',
                    r'defend,?\s+indemnif(y|ies|ied)',
                    r'hold\s+.*?harmless\s+from\s+and?\s+against',
                ],
                'risk_level': 'high',
                'explanation': 'Indemnification clauses allocate liability between parties and may expose one party to significant financial risk.',
                'recommendations': [
                    'Consider mutual indemnification',
                    'Include reasonable limitations on liability',
                    'Specify indemnification procedures',
                    'Include carve-outs for willful misconduct'
                ]
            },
            'limitation_of_liability': {
                'patterns': [
                    r'limitation\s+of\s+liability',
                    r'exclu(sion|de)\s+of\s+(consequential|incidental|special|indirect)',
                    r'not\s+liable\s+for\s+(consequential|incidental|special|indirect)',
                    r'liability\s+cap',
                    r'aggregate\s+liability',
                ],
                'risk_level': 'medium',
                'explanation': 'Limitation of liability clauses cap potential damages and may limit legal remedies.',
                'recommendations': [
                    'Ensure mutual limitations',
                    'Exclude liability for gross negligence/willful misconduct',
                    'Consider industry-standard caps',
                    'Include carve-outs for IP infringement'
                ]
            },
            'termination': {
                'patterns': [
                    r'terminat(e|ion|ing)(\s+for\s+convenience)?',
                    r'right\s+to\s+terminate',
                    r'termination\s+rights',
                    r'early\s+termination',
                ],
                'risk_level': 'medium',
                'explanation': 'Termination clauses define the conditions under which parties can exit the agreement.',
                'recommendations': [
                    'Include cure periods for material breaches',
                    'Specify notice requirements',
                    'Define post-termination obligations',
                    'Address survival of key provisions'
                ]
            },
            'governing_law': {
                'patterns': [
                    r'govern(ing)?\s+law',
                    r'jurisdiction',
                    r'venue',
                    r'dispute\s+resolution',
                ],
                'risk_level': 'low',
                'explanation': 'Governing law and jurisdiction clauses determine which laws apply and where disputes will be resolved.',
                'recommendations': [
                    'Specify neutral jurisdiction',
                    'Consider arbitration for international contracts',
                    'Include waiver of jury trial if appropriate',
                    'Specify language for dispute resolution'
                ]
            },
            'intellectual_property': {
                'patterns': [
                    r'intellectual\s+property',
                    r'ip\s+rights',
                    r'ownership\s+of\s+work\s+product',
                    r'licen[cs]e',
                    r'patent',
                    r'copyright',
                    r'trademark',
                ],
                'risk_level': 'high',
                'explanation': 'IP clauses define ownership and usage rights of intellectual property.',
                'recommendations': [
                    'Clearly define IP ownership',
                    'Include appropriate license grants',
                    'Address background and foreground IP',
                    'Include IP indemnification'
                ]
            },
            'confidentiality': {
                'patterns': [
                    r'confidential',
                    r'non[\s-]?disclosure',
                    r'nda',
                    r'trade\s+secret',
                ],
                'risk_level': 'medium',
                'explanation': 'Confidentiality provisions protect sensitive business information.',
                'recommendations': [
                    'Define confidential information clearly',
                    'Include exceptions to confidentiality',
                    'Specify duration of obligations',
                    'Address permitted disclosures'
                ]
            },
            'force_majeure': {
                'patterns': [
                    r'force\s+majeure',
                    r'act\s+of\s+god',
                    r'beyond\s+reasonable\s+control',
                ],
                'risk_level': 'medium',
                'explanation': 'Force majeure clauses address unforeseeable circumstances that prevent contract performance.',
                'recommendations': [
                    'Define force majeure events specifically',
                    'Include notice requirements',
                    'Address right to terminate for prolonged force majeure',
                    'Consider pandemic-specific language'
                ]
            },
            'assignment': {
                'patterns': [
                    r'assign(ment)?',
                    r'change\s+of\s+control',
                    r'successors?\s+and\s+assigns',
                ],
                'risk_level': 'medium',
                'explanation': 'Assignment clauses govern whether and how parties can transfer their rights and obligations.',
                'recommendations': [
                    'Specify permitted assignments',
                    'Include change of control provisions',
                    'Require written consent for assignments',
                    'Address assignment in case of merger/acquisition'
                ]
            },
            'warranties': {
                'patterns': [
                    r'warrant(y|ies)',
                    r'representation',
                    r'disclaimer',
                    r'as\s+is',
                ],
                'risk_level': 'high',
                'explanation': 'Warranties are promises about the quality or characteristics of goods/services.',
                'recommendations': [
                    'Ensure warranties are accurate and supportable',
                    'Include appropriate disclaimers',
                    'Consider industry-specific warranties',
                    'Address remedies for breach of warranty'
                ]
            },
            'amendment': {
                'patterns': [
                    r'amend(ment)?',
                    r'modif(y|ication)',
                    r'change',
                    r'waiver',
                ],
                'risk_level': 'low',
                'explanation': 'Amendment clauses specify how the agreement can be modified.',
                'recommendations': [
                    'Require written amendments',
                    'Specify required signatories',
                    'Address electronic signatures',
                    'Consider no-oral-modification clauses'
                ]
            },
            'notices': {
                'patterns': [
                    r'notice',
                    r'communication',
                    r'contact',
                ],
                'risk_level': 'low',
                'explanation': 'Notice provisions specify how and where formal communications should be sent.',
                'recommendations': [
                    'Specify notice methods (email, certified mail, etc.)',
                    'Include notice addresses',
                    'Address when notice is deemed received',
                    'Consider electronic notice provisions'
                ]
            },
            'severability': {
                'patterns': [
                    r'severability',
                    r'invalidity',
                    r'unenforceable',
                ],
                'risk_level': 'low',
                'explanation': 'Severability clauses ensure that if one part of the agreement is invalid, the rest remains in effect.',
                'recommendations': [
                    'Include a severability clause',
                    'Consider a reformation provision',
                    'Address blue-pencil doctrine if applicable',
                    'Include a savings clause'
                ]
            },
            'entire_agreement': {
                'patterns': [
                    r'entire\s+agreement',
                    r'integration',
                    r'supersede',
                ],
                'risk_level': 'low',
                'explanation': 'The entire agreement clause states that the written contract represents the complete understanding between the parties.',
                'recommendations': [
                    'Include an integration clause',
                    'Address prior agreements',
                    'Consider carve-outs for side letters',
                    'Address reliance on representations'
                ]
            },
            'counterparts': {
                'patterns': [
                    r'counterpart',
                    r'multiple\s+originals',
                    r'electronic\s+signature',
                ],
                'risk_level': 'low',
                'explanation': 'Counterparts clauses allow the agreement to be signed in multiple copies, each of which is considered an original.',
                'recommendations': [
                    'Include a counterparts clause',
                    'Address electronic signatures',
                    'Specify delivery methods',
                    'Consider notarization requirements if applicable'
                ]
            },
            'survival': {
                'patterns': [
                    r'surviv(e|al)',
                    r'termination',
                    r'expiration',
                ],
                'risk_level': 'medium',
                'explanation': 'Survival clauses specify which obligations continue after the agreement ends.',
                'recommendations': [
                    'List provisions that survive termination',
                    'Consider survival of indemnification obligations',
                    'Address return/destruction of confidential information',
                    'Consider survival periods for different obligations'
                ]
            },
            'waiver': {
                'patterns': [
                    r'waiv(e|er|ing)',
                    r'no\s+waiver',
                    r'failure\s+to\s+enforce',
                ],
                'risk_level': 'low',
                'explanation': 'Waiver provisions address whether a party waives its rights by not enforcing them.',
                'recommendations': [
                    'Include a no-waiver clause',
                    'Specify that waivers must be in writing',
                    'Address course of dealing',
                    'Consider estoppel implications'
                ]
            },
            'relationship': {
                'patterns': [
                    r'relationship\s+of\s+the\s+parties',
                    r'independent\s+contractor',
                    r'no\s+partnership',
                    r'no\s+joint\s+venture',
                ],
                'risk_level': 'low',
                'explanation': 'Relationship clauses define the legal relationship between the parties.',
                'recommendations': [
                    'Specify independent contractor status if applicable',
                    'Disclaim partnership or joint venture',
                    'Address tax implications',
                    'Consider agency relationships'
                ]
            },
            'publicity': {
                'patterns': [
                    r'publicity',
                    r'press\s+release',
                    r'marketing',
                    r'use\s+of\s+(name|logo)',
                ],
                'risk_level': 'low',
                'explanation': 'Publicity clauses govern whether and how parties can mention their relationship.',
                'recommendations': [
                    'Specify approval rights for press releases',
                    'Address use of trademarks/logos',
                    'Include confidentiality for terms',
                    'Consider social media mentions'
                ]
            },
            'audit': {
                'patterns': [
                    r'audit',
                    r'inspection',
                    r'records',
                    r'books',
                ],
                'risk_level': 'medium',
                'explanation': 'Audit rights allow a party to verify compliance with the agreement.',
                'recommendations': [
                    'Define scope of audit rights',
                    'Include notice requirements',
                    'Limit frequency of audits',
                    'Address confidentiality of audit results'
                ]
            },
            'insurance': {
                'patterns': [
                    r'insurance',
                    r'coverage',
                    r'policy',
                ],
                'risk_level': 'medium',
                'explanation': 'Insurance clauses require parties to maintain certain types of insurance coverage.',
                'recommendations': [
                    'Specify required insurance types and limits',
                    'Require additional insured status if needed',
                    'Address notice of cancellation',
                    'Consider industry-specific requirements'
                ]
            },
            'export_control': {
                'patterns': [
                    r'export\s+control',
                    r'ear',
                    r'itar',
                    r'ofac',
                ],
                'risk_level': 'high',
                'explanation': 'Export control clauses ensure compliance with international trade regulations.',
                'recommendations': [
                    'Include export control representations',
                    'Address restricted party screening',
                    'Consider technology control plans',
                    'Specify compliance with applicable laws'
                ]
            },
            'compliance': {
                'patterns': [
                    r'compliance',
                    r'law',
                    r'regulation',
                    r'statute',
                ],
                'risk_level': 'high',
                'explanation': 'Compliance clauses require parties to adhere to applicable laws and regulations.',
                'recommendations': [
                    'Include general compliance obligations',
                    'Address specific regulatory requirements',
                    'Consider anti-corruption laws (FCPA, UKBA, etc.)',
                    'Include data protection/GDPR provisions if applicable'
                ]
            },
            'dispute_resolution': {
                'patterns': [
                    r'dispute\s+resolution',
                    r'arbitration',
                    r'litigation',
                    r'jurisdiction',
                ],
                'risk_level': 'high',
                'explanation': 'Dispute resolution clauses specify how disputes will be resolved.',
                'recommendations': [
                    'Consider mediation before arbitration/litigation',
                    'Specify governing law and venue',
                    'Address injunctive relief',
                    'Consider class action waivers'
                ]
            },
            'attorneys_fees': {
                'patterns': [
                    r'attorney[\'\s]s?\s+fees',
                    r'legal\s+fees',
                    r'costs',
                ],
                'risk_level': 'medium',
                'explanation': 'Attorneys\' fees provisions determine whether the prevailing party can recover legal costs.',
                'recommendations': [
                    'Consider mutual attorneys\' fees provisions',
                    'Define "prevailing party"',
                    'Address costs beyond attorneys\' fees',
                    'Consider fee-shifting statutes'
                ]
            },
            'no_third_party_beneficiaries': {
                'patterns': [
                    r'third[\s-]party\s+beneficiar(y|ies)',
                    r'no\s+third[\s-]party\s+rights',
                ],
                'risk_level': 'low',
                'explanation': 'These clauses prevent third parties from enforcing the agreement.',
                'recommendations': [
                    'Include a no-third-party-beneficiaries clause',
                    'Consider carve-outs if third-party rights are intended',
                    'Address assignees and successors',
                    'Consider indemnified parties'
                ]
            },
            'headings': {
                'patterns': [
                    r'headings',
                    r'table\s+of\s+contents',
                    r'section\s+headings',
                ],
                'risk_level': 'low',
                'explanation': 'Headings clauses state that section headings are for convenience only.',
                'recommendations': [
                    'Include a headings clause',
                    'Address construction of the agreement',
                    'Consider definitions sections',
                    'Address use of examples'
                ]
            }
        }

    def analyze_document(self, text: str) -> List[RiskClause]:
        """Analyze document for risky clauses"""
        logger.info("Analyzing document for risky clauses...")
        
        risky_clauses = []
        text_lower = text.lower()
        
        # Split into sentences for better context
        sentences = nltk.sent_tokenize(text)
        
        for clause_type, clause_info in self.legal_terms.items():
            for pattern in clause_info["patterns"]:
                matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                
                for match in matches:
                    # Find the sentence containing this match
                    start_pos = match.start()
                    context = self._get_context(text, start_pos, sentences)
                    
                    # Get location information
                    location_info = self._estimate_page_location(text, start_pos)
                    
                    risk_clause = RiskClause(
                        clause_type=clause_type.replace("_", " ").title(),
                        risk_level=clause_info["risk_level"],
                        text=context,
                        location=start_pos,
                        explanation=clause_info["explanation"],
                        recommendations=clause_info["recommendations"]
                    )
                    
                    # Add location information as attributes
                    risk_clause.page_location = location_info["estimated_page"]
                    risk_clause.section_location = location_info["section"]
                    risky_clauses.append(risk_clause)
        
        # Remove duplicates and sort by location
        risky_clauses = self._deduplicate_clauses(risky_clauses)
        risky_clauses.sort(key=lambda x: x.location)
        
        logger.info(f"Found {len(risky_clauses)} risky clauses")
        return risky_clauses
    
    def _get_context(self, text: str, position: int, sentences: List[str]) -> str:
        """Get the context (full sentence or paragraph) for a matched position"""
        # Find which sentence contains this position
        current_pos = 0
        for sentence in sentences:
            sentence_start = text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)
            
            if sentence_start <= position <= sentence_end:
                return sentence.strip()
            
            current_pos = sentence_end
        
        # Fallback: return surrounding text
        start = max(0, position - 100)
        end = min(len(text), position + 200)
        return text[start:end].strip()
    
    def _estimate_page_location(self, text: str, position: int) -> dict:
        """Estimate page and section location of the risk clause"""
        # Count characters per page (rough estimate: 2000 chars per page)
        chars_per_page = 2000
        estimated_page = (position // chars_per_page) + 1
        
        # Find section context by looking for headers or numbered sections
        text_before = text[:position].lower()
        
        # Look for section markers
        section_patterns = [
            r'section\s+(\d+)',
            r'article\s+(\d+)',
            r'clause\s+(\d+)',
            r'paragraph\s+(\d+)',
            r'(\d+)\.\s*[A-Z]'
        ]
        
        section_info = "Unknown Section"
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text_before))
            if matches:
                last_match = matches[-1]
                section_info = f"Section {last_match.group(1)}"
                break
        
        return {
            "estimated_page": estimated_page,
            "section": section_info,
            "character_position": position
        }
    
    def _deduplicate_clauses(self, clauses: List[RiskClause]) -> List[RiskClause]:
        """Remove duplicate or overlapping risk clauses"""
        if not clauses:
            return []
        
        # Sort by location
        clauses.sort(key=lambda x: x.location)
        
        deduplicated = [clauses[0]]
        for clause in clauses[1:]:
            # Check if this clause overlaps with the previous one
            last_clause = deduplicated[-1]
            if (clause.location - last_clause.location > 50 or 
                clause.clause_type != last_clause.clause_type):
                deduplicated.append(clause)
        
        return deduplicated
    
    def calculate_risk_score(self, risky_clauses: List[RiskClause]) -> Dict[str, any]:
        """Calculate overall risk score for the document"""
        if not risky_clauses:
            return {
                "overall_score": 0,
                "risk_level": "low",
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "total_risks": 0
            }
        
        risk_counts = defaultdict(int)
        
        for clause in risky_clauses:
            risk_counts[clause.risk_level] += 1
        
        # Calculate score based on actual risk distribution
        high_count = risk_counts["high"]
        medium_count = risk_counts["medium"]
        low_count = risk_counts["low"]
        
        # Score calculation: High=30 points, Medium=15 points, Low=5 points each
        score = (high_count * 30) + (medium_count * 15) + (low_count * 5)
        
        # Determine overall risk level based on highest severity present
        if high_count > 0:
            overall_risk = "high"
        elif medium_count > 0:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_score": min(score, 100),  # Cap at 100
            "risk_level": overall_risk,
            "high_risk_count": high_count,
            "medium_risk_count": medium_count,
            "low_risk_count": low_count,
            "total_risks": len(risky_clauses)
        }
    
    def generate_summary_report(self, risky_clauses: List[RiskClause], risk_score: Dict) -> str:
        """Generate a summary report of the analysis"""
        report = f"""
# Legal Document Risk Analysis Report

## Overall Risk Assessment
- **Risk Score**: {risk_score['overall_score']}/100
- **Risk Level**: {risk_score['risk_level'].upper()}
- **Total Risky Clauses Found**: {risk_score['total_risks']}
  - High Risk: {risk_score['high_risk_count']}
  - Medium Risk: {risk_score['medium_risk_count']}
  - Low Risk: {risk_score['low_risk_count']}

## Detailed Findings

"""
        # Group clauses by risk level
        for risk_level in ["high", "medium", "low"]:
            level_clauses = [c for c in risky_clauses if c.risk_level == risk_level]
            if level_clauses:
                report += f"### {risk_level.upper()} RISK CLAUSES\n\n"
                for i, clause in enumerate(level_clauses, 1):
                    report += f"""
**{i}. {clause.clause_type}**
- **Risk Level**: {clause.risk_level.upper()}
- **Explanation**: {clause.explanation}
- **Found Text**: "{clause.text[:200]}..."
- **Recommendations**:
"""
                    for rec in clause.recommendations:
                        report += f"  - {rec}\n"
                    report += "\n"
        
        return report
