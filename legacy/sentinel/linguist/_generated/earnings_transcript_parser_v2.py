"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for downstream LLM analysis via Linguist. Handles
common transcript formats (Seeking Alpha, investor relations platforms) and
normalizes speaker labels to canonical roles. Output segments feed into
sample_score.py for certainty and hesitation analysis per speaker.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum


class SpeakerRole(Enum):
    """Canonical speaker role classification."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    ANALYST = "Analyst"
    OPERATOR = "Operator"
    OTHER = "Other"


@dataclass
class TranscriptSegment:
    """A single speaker's utterance within a transcript."""
    speaker_name: str
    speaker_role: SpeakerRole
    timestamp: Optional[str]
    text: str
    segment_index: int


@dataclass
class ParsedTranscript:
    """Complete parsed earnings call transcript with metadata."""
    company_ticker: str
    call_date: Optional[str]
    call_type: str  # "earnings", "conference", "shareholder"
    segments: List[TranscriptSegment]
    raw_text: str


def classify_speaker_role(speaker_name: str, title_hint: Optional[str] = None) -> SpeakerRole:
    """
    Classify a speaker into canonical role based on name and optional title.
    
    Applies heuristic matching against common executive titles and analyst keywords.
    Falls back to OTHER if no match found.
    """
    combined = f"{speaker_name} {title_hint or ''}".lower()
    
    role_patterns = {
        SpeakerRole.CEO: [r"\bceo\b", r"\bchief executive", r"chief exec"],
        SpeakerRole.CFO: [r"\bcfo\b", r"\bchief financial", r"\btreasurer\b"],
        SpeakerRole.COO: [r"\bcoo\b", r"\bchief operating"],
        SpeakerRole.CTO: [r"\bcto\b", r"\bchief technology"],
        SpeakerRole.ANALYST: [r"\banalyst\b", r"\bequity research", r"\bresearch analyst"],
        SpeakerRole.OPERATOR: [r"\boperator\b", r"\bfacilitator\b", r"\bhost\b"],
    }
    
    for role, patterns in role_patterns.items():
        for pattern in patterns:
            if re.search(pattern, combined):
                return role
    
    return SpeakerRole.OTHER


def extract_timestamp(line: str) -> Optional[str]:
    """
    Extract optional timestamp from transcript line (e.g., "[00:12:34]").
    
    Returns formatted timestamp string or None if not found.
    """
    match = re.search(r"\[(\d{1,2}:\d{2}:\d{2})\]", line)
    return match.group(1) if match else None


def parse_seeking_alpha_format(transcript_text: str, company_ticker: str) -> ParsedTranscript:
    """
    Parse Seeking Alpha earnings call transcript format.
    
    Expected format:
      Company Name Earnings Call
      [timestamp optional]
      Speaker Name, Title
      Speaker text... paragraph breaks separate utterances.
    """
    lines = transcript_text.split('\n')
    segments: List[TranscriptSegment] = []
    
    current_speaker_name: Optional[str] = None
    current_speaker_role: SpeakerRole = SpeakerRole.OTHER
    current_timestamp: Optional[str] = None
    current_text_buffer: List[str] = []
    segment_index = 0
    
    call_date: Optional[str] = None
    call_type = "earnings"
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # Try to extract date from header
        if call_date is None and i < 5:
            date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})", line)
            if date_match:
                call_date = date_match.group(1)
        
        # Extract timestamp if present
        timestamp = extract_timestamp(line)
        if timestamp:
            current_timestamp = timestamp
            line = re.sub(r"\[\d{1,2}:\d{2}:\d{2}\]", "", line).strip()
        
        # Detect speaker line: typically "Name, Title" or just "Name"
        # Heuristic: line is short (< 100 chars), contains comma or ALL_CAPS name
        if line and len(line) < 100 and (
            re.match(r"^[A-Z][a-z\s'-]+(?:,|$)", line) or
            re.search(r"\b(?:CEO|CFO|COO|CTO|Analyst|Operator)\b", line)
        ):
            # Flush previous speaker segment
            if current_speaker_name and current_text_buffer:
                text = ' '.join(current_text_buffer).strip()
                if text:
                    segments.append(TranscriptSegment(
                        speaker_name=current_speaker_name,
                        speaker_role=current_speaker_role,
                        timestamp=current_timestamp,
                        text=text,
                        segment_index=segment_index
                    ))
                    segment_index += 1
            
            # Parse new speaker
            parts = [p.strip() for p in line.split(',', 1)]
            current_speaker_name = parts[0]
            title_hint = parts[1] if len(parts) > 1 else None
            current_speaker_role = classify_speaker_role(current_speaker_name, title_hint)
            current_text_buffer = []
        
        elif line and current_speaker_name:
            # Accumulate text for current speaker
            current_text_buffer.append(line)
    
    # Flush final speaker
    if current_speaker_name and current_text_buffer:
        text = ' '.join(current_text_buffer).strip()
        if text:
            segments.append(TranscriptSegment(
                speaker_name=current_speaker_name,
                speaker_role=current_speaker_role,
                timestamp=current_timestamp,
                text=text,
                segment_index=segment_index
            ))
    
    return ParsedTranscript(
        company_ticker=company_ticker,
        call_date=call_date,
        call_type=call_type,
        segments=segments,
        raw_text=transcript_text
    )


def parse_generic_format(transcript_text: str, company_ticker: str) -> ParsedTranscript:
    """
    Parse generic transcript format with flexible speaker detection.
    
    Looks for lines starting with speaker name (capitalized) followed by colon or newline,
    then accumulates utterance text until next speaker line.
    """
    lines = transcript_text.split('\n')
    segments: List[TranscriptSegment] = []
    
    current_speaker_name: Optional[str] = None
    current_speaker_role: SpeakerRole = SpeakerRole.OTHER
    current_timestamp: Optional[str] = None
    current_text_buffer: List[str] = []
    segment_index = 0
    
    call_date: Optional[str] = None
    
    for line in lines:
        line = line.rstrip()
        
        # Extract date if present
        if call_date is None:
            date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})", line)
            if date_match:
                call_date = date_match.group(1)
