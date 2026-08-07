"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, etc.)
and prepares each segment for LLM sentiment analysis. Handles common transcript
formats (seeking alpha, company IR pages, etc.) and normalizes speaker
classifications for downstream Linguist analysis.

Used by linguist pipeline to extract role-specific tone and certainty signals
before passing to Claude for nuanced reasoning.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple


class SpeakerRole(Enum):
    """Enumeration of speaker roles in earnings calls."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    ANALYST = "Analyst"
    OPERATOR = "Operator"
    UNKNOWN = "Unknown"


@dataclass
class TranscriptSegment:
    """A single speaker's turn in an earnings call."""
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    line_number: int
    is_qa_section: bool = False


@dataclass
class ParsedTranscript:
    """Structured representation of an earnings call transcript."""
    company_ticker: Optional[str]
    call_date: Optional[str]
    segments: List[TranscriptSegment]
    raw_text: str
    metadata: Dict[str, str]


def infer_speaker_role(speaker_name: str, context: Optional[str] = None) -> SpeakerRole:
    """
    Infer speaker role from name and optional context string.
    
    Uses heuristics: titles in parentheses, common name patterns,
    and context clues. Falls back to UNKNOWN.
    """
    name_lower = speaker_name.lower()
    context_lower = (context or "").lower()
    
    # Direct title matching in name
    if any(title in name_lower for title in ["chief executive", "ceo", "president & ceo"]):
        return SpeakerRole.CEO
    if any(title in name_lower for title in ["chief financial", "cfo", "finance"]):
        return SpeakerRole.CFO
    if any(title in name_lower for title in ["chief operating", "coo", "operations"]):
        return SpeakerRole.COO
    if any(title in name_lower for title in ["chief technology", "cto", "technology"]):
        return SpeakerRole.CTO
    
    # Operator detection
    if "operator" in name_lower or "moderator" in name_lower:
        return SpeakerRole.OPERATOR
    
    # Analyst heuristics
    if any(bank in name_lower for bank in ["goldman", "morgan", "jpmorgan", "barclays", 
                                             "citi", "bank of america", "goldman sachs",
                                             "morgan stanley"]):
        return SpeakerRole.ANALYST
    if "analyst" in name_lower or "research" in name_lower:
        return SpeakerRole.ANALYST
    
    # Context clues in transcript
    if context_lower and any(phrase in context_lower for phrase in [
        "from the investor relations",
        "investor question",
        "research analyst",
        "equity analyst"
    ]):
        return SpeakerRole.ANALYST
    
    return SpeakerRole.UNKNOWN


def detect_qa_boundary(text: str) -> bool:
    """
    Detect if text marks the start of Q&A section.
    
    Returns True if common Q&A delimiters or section headers are found.
    """
    qa_markers = [
        r"^(question and answer|q\s*&\s*a|qa session)",
        r"^(operator:?\s+)?(we\s+)?will\s+now\s+(take|open|begin|move to)\s+(the\s+)?questions?",
        r"^(now\s+)?(let'?s\s+)?open\s+(it\s+)?up\s+to\s+questions?",
    ]
    text_lower = text.strip().lower()
    return any(re.search(pattern, text_lower, re.MULTILINE) for pattern in qa_markers)


def parse_transcript(text: str, company_ticker: Optional[str] = None,
                     call_date: Optional[str] = None) -> ParsedTranscript:
    """
    Parse earnings call transcript into segments by speaker role.
    
    Handles common formats with speaker names followed by colons or dashes.
    Normalizes whitespace and detects Q&A section boundary.
    Returns ParsedTranscript with classified segments.
    """
    if not text or not isinstance(text, str):
        return ParsedTranscript(
            company_ticker=company_ticker,
            call_date=call_date,
            segments=[],
            raw_text=text or "",
            metadata={}
        )
    
    segments: List[TranscriptSegment] = []
    is_qa_section = False
    
    # Split on common speaker patterns: "NAME:" or "NAME —" or "NAME|"
    speaker_pattern = r"^([A-Z][A-Za-z\s\.\,\&\']+?)[\:\—\|]"
    
    lines = text.split("\n")
    current_line_num = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        current_line_num += 1
        
        # Check for Q&A boundary
        if detect_qa_boundary(line):
            is_qa_section = True
            i += 1
            continue
        
        # Try to match speaker line
        match = re.match(speaker_pattern, line.strip())
        if match:
            speaker_name = match.group(1).strip()
            
            # Collect speaker's text (may span multiple lines)
            speaker_text_parts = [line[match.end():].strip()]
            
            # Look ahead for continuation lines (not starting with speaker pattern)
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                if re.match(speaker_pattern, next_line):
                    break
                speaker_text_parts.append(next_line)
                current_line_num += 1
                i += 1
            
            speaker_text = " ".join(speaker_text_parts).strip()
            
            # Infer role
            speaker_role = infer_speaker_role(speaker_name, speaker_text[:100])
            
            if speaker_text:
                segment = TranscriptSegment(
                    speaker_name=speaker_name,
                    speaker_role=speaker_role,
                    text=speaker_text,
                    line_number=current_line_num,
                    is_qa_section=is_qa_section
                )
                segments.append(segment)
        else:
            i += 1
    
    return ParsedTranscript(
        company_ticker=company_ticker,
        call_date=call_date,
        segments=segments,
        raw_text=text,
        metadata={"total_lines": len(lines), "qa_detected": is_qa_section}
    )


def extract_by_role(parsed: ParsedTranscript, 
                   role: SpeakerRole) -> List[TranscriptSegment]:
    """
    Extract all segments for a specific speaker role from parsed transcript.
    """
    return [seg for seg in parsed.segments if seg.speaker_role == role]


def extract_qa_segments(parsed: ParsedTranscript) -> List[TranscriptSegment]:
    """
    Extract only Q&A section segments from parsed transcript.
    """
    return [seg for seg in parsed.segments if seg.is_qa_section]


def extract_prepared_remarks(parsed: ParsedTranscript) -> List[TranscriptSegment]:
    """
    Extract only prepared remarks (non-Q&A) segments from parsed transcript.
    """
    return [seg for seg in parsed.segments if
