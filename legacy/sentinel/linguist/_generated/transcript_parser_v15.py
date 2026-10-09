"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments raw earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for downstream LLM sentiment analysis via the Linguist pillar.
Extracts speaker metadata, normalizes formatting, and produces structured JSON output
suitable for Claude reasoning on tone drift and regulatory signals.

Integrates with sentinel/linguist/sample_score.py for per-segment certainty scoring.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum


class SpeakerRole(str, Enum):
    """Enumeration of speaker roles in earnings calls."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    ANALYST = "ANALYST"
    OPERATOR = "OPERATOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class Segment:
    """A single speaker segment from an earnings call transcript."""
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    line_number: int
    company_ticker: Optional[str] = None
    call_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert segment to dictionary."""
        return {
            **asdict(self),
            "speaker_role": self.speaker_role.value
        }


@dataclass
class TranscriptMetadata:
    """Metadata extracted from earnings call transcript header."""
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    call_date: Optional[str] = None
    quarter: Optional[str] = None
    fiscal_year: Optional[int] = None
    source_url: Optional[str] = None


def _detect_speaker_role(name: str, text_sample: str = "") -> SpeakerRole:
    """
    Infer speaker role from name and optional text sample.
    
    Heuristics: matches against title keywords (CEO, CFO, etc.),
    considers analyst question patterns, defaults to ANALYST if uncertain,
    OPERATOR if matches call-running keywords.
    """
    name_lower = name.lower()
    
    # Title-based detection.
    if re.search(r'\bceo\b|\bchief executive\b', name_lower):
        return SpeakerRole.CEO
    if re.search(r'\bcfo\b|\bchief financial\b', name_lower):
        return SpeakerRole.CFO
    if re.search(r'\bcoo\b|\bchief operating\b', name_lower):
        return SpeakerRole.COO
    if re.search(r'\bcto\b|\bchief technology\b|\bhead of engineering\b', name_lower):
        return SpeakerRole.CTO
    
    # Operator keywords (call facilitation).
    if re.search(r'\boperator\b|\bfacilitator\b|\bhost\b', name_lower):
        return SpeakerRole.OPERATOR
    
    # Analyst heuristics from text (e.g., "Thanks for the call" or "question" patterns).
    if text_sample:
        text_lower = text_sample[:200].lower()
        if re.search(r'\bquestion\b|\bcan you please\b|\bwould you comment\b', text_lower):
            return SpeakerRole.ANALYST
    
    # Default: assume analyst if name suggests external party, else unknown.
    if any(keyword in name_lower for keyword in ['analyst', 'morgan', 'goldman', 'jpmorgan', 'bank', 'research']):
        return SpeakerRole.ANALYST
    
    return SpeakerRole.UNKNOWN


def extract_metadata(transcript_text: str) -> TranscriptMetadata:
    """
    Extract call metadata (company, ticker, date, quarter, etc.) from transcript header.
    
    Looks for common header patterns: "Company: X", "Ticker: Y", date formats (YYYY-MM-DD, Month DD, YYYY),
    and quarter references (Q1/Q2/Q3/Q4).
    """
    metadata = TranscriptMetadata()
    
    # Company name extraction.
    company_match = re.search(r'(?:company|corporation):\s*([A-Za-z\s&.,]+?)(?:\n|ticker)', transcript_text, re.IGNORECASE)
    if company_match:
        metadata.company_name = company_match.group(1).strip()
    
    # Ticker extraction.
    ticker_match = re.search(r'(?:ticker|symbol|nyse|nasdaq):\s*([A-Z]{1,5})', transcript_text, re.IGNORECASE)
    if ticker_match:
        metadata.ticker = ticker_match.group(1).strip()
    
    # Date extraction (multiple formats).
    date_patterns = [
        r'(?:date|call date):\s*(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(?:date|call date):\s*(\w+\s+\d{1,2},\s*\d{4})',  # Month DD, YYYY
        r'(\d{4}-\d{2}-\d{2})',  # Bare YYYY-MM-DD in header
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, transcript_text[:500], re.IGNORECASE)
        if date_match:
            metadata.call_date = date_match.group(1).strip()
            break
    
    # Quarter and fiscal year extraction.
    quarter_match = re.search(r'(Q[1-4])\s*(\d{4})?', transcript_text[:300], re.IGNORECASE)
    if quarter_match:
        metadata.quarter = quarter_match.group(1).upper()
        if quarter_match.group(2):
            metadata.fiscal_year = int(quarter_match.group(2))
    
    return metadata


def parse_transcript(transcript_text: str, ticker: Optional[str] = None, call_date: Optional[str] = None) -> Tuple[List[Segment], TranscriptMetadata]:
    """
    Parse earnings call transcript into speaker segments.
    
    Detects speaker boundaries (patterns like "Speaker Name:" or "SPEAKER NAME"),
    infers roles, normalizes whitespace, and returns list of Segment objects
    plus extracted metadata. Handles multiline speaker introductions and Q&A sections.
    """
    metadata = extract_metadata(transcript_text)
    if ticker:
        metadata.ticker = ticker
    if call_date:
        metadata.call_date = call_date
    
    segments: List[Segment] = []
    
    # Split into lines for processing.
    lines = transcript_text.split('\n')
    
    # Pattern for speaker line: "SPEAKER NAME:" or "Speaker Name:" or "SPEAKER NAME —" or similar.
    speaker_pattern = re.compile(
        r'^[*\s]*([A-Za-z\s,\.&\-()]+?)\s*(?::|—|-)\s*(.*)$',
        re.MULTILINE
    )
    
    current_segment_lines: List[str] = []
    current_speaker: Optional[str] = None
    current_line_num = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for speaker line.
        match = speaker_pattern.match(line.strip())
        if match and len(line.strip()) > 2:
            speaker_name = match.group(1).strip()
            first_text = match.group(2).strip() if match.group(2) else ""
            
            # Validate speaker name (reject common false positives).
            if (
                len(speaker_name) > 2 and
                len(speaker_name) < 100 and
                not re.match(r'^\d+\.|^\(.*\)$', speaker_name) and
