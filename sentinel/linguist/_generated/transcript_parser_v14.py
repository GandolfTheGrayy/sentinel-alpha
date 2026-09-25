"""
Earnings call transcript parser for Sentinel Linguist.

Segments raw earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for downstream LLM sentiment analysis. Handles common
transcript formats (Seeking Alpha, company IR pages, SEC EDGAR).

Used by Linguist to isolate management tone vs. analyst skepticism signals.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class SpeakerRole(Enum):
    """Enumeration of speaker roles in earnings call transcripts."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    ANALYST = "ANALYST"
    OPERATOR = "OPERATOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class TranscriptSegment:
    """A single speaker utterance within a transcript."""
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    timestamp_seconds: Optional[int] = None
    line_number: int = 0


@dataclass
class ParsedTranscript:
    """Parsed earnings call transcript with segmented speaker utterances."""
    ticker: str
    company_name: str
    call_date: Optional[str]
    segments: List[TranscriptSegment]
    raw_text: str


def infer_speaker_role(name: str) -> SpeakerRole:
    """Infer speaker role from name and title patterns."""
    name_lower = name.lower()
    
    if re.search(r'\bceo\b|\bchief executive\b', name_lower):
        return SpeakerRole.CEO
    elif re.search(r'\bcfo\b|\bchief financial\b|\btreasurer\b', name_lower):
        return SpeakerRole.CFO
    elif re.search(r'\bcoo\b|\bchief operating\b', name_lower):
        return SpeakerRole.COO
    elif re.search(r'\bcto\b|\bchief technology\b', name_lower):
        return SpeakerRole.CTO
    elif re.search(r'\banalyst\b|\bequity research\b|\bquestion\b', name_lower):
        return SpeakerRole.ANALYST
    elif re.search(r'\boperator\b', name_lower):
        return SpeakerRole.OPERATOR
    else:
        return SpeakerRole.UNKNOWN


def parse_seeking_alpha_format(text: str, ticker: str, company_name: str, call_date: Optional[str] = None) -> ParsedTranscript:
    """
    Parse Seeking Alpha earnings call transcript format.
    
    Expects lines like:
      John Doe
      CEO
      Text of statement...
    """
    segments: List[TranscriptSegment] = []
    lines = text.split('\n')
    
    i = 0
    line_number = 0
    while i < len(lines):
        line = lines[i].strip()
        line_number += 1
        
        # Skip empty lines and common headers
        if not line or line.startswith('---') or line.startswith('=='):
            i += 1
            continue
        
        # Check if this looks like a speaker name (next non-empty line should be role or text)
        speaker_name = line
        i += 1
        
        # Try to find role on next line
        speaker_role = SpeakerRole.UNKNOWN
        if i < len(lines):
            next_line = lines[i].strip().upper()
            if next_line in ('CEO', 'CFO', 'COO', 'CTO', 'ANALYST', 'OPERATOR'):
                speaker_role = SpeakerRole[next_line]
                i += 1
            else:
                speaker_role = infer_speaker_role(speaker_name)
        
        # Collect speaker text until next speaker
        text_lines: List[str] = []
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate:
                i += 1
                continue
            
            # Simple heuristic: if line is SHORT and FOLLOWED by another short line,
            # likely a new speaker block
            if (len(candidate) < 80 and 
                i + 1 < len(lines) and 
                len(lines[i + 1].strip()) < 80 and
                re.match(r'^[A-Z][a-z]+\s+[A-Z]', candidate)):
                break
            
            text_lines.append(candidate)
            i += 1
        
        segment_text = ' '.join(text_lines).strip()
        if segment_text:
            segments.append(TranscriptSegment(
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                text=segment_text,
                line_number=line_number
            ))
    
    return ParsedTranscript(
        ticker=ticker,
        company_name=company_name,
        call_date=call_date,
        segments=segments,
        raw_text=text
    )


def parse_generic_format(text: str, ticker: str, company_name: str, call_date: Optional[str] = None) -> ParsedTranscript:
    """
    Parse generic 'Speaker Name: text...' format.
    
    Handles patterns like:
      John Doe, CEO: Here is my statement...
      Analyst Name: What about costs?
    """
    segments: List[TranscriptSegment] = []
    
    # Pattern: "Name [, Title]: Text"
    pattern = r'^([^:]+?)(?:,\s*([^:]+?))?\s*:\s*(.+)$'
    
    for line_number, line in enumerate(text.split('\n'), 1):
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern, line)
        if match:
            speaker_name = match.group(1).strip()
            title_hint = match.group(2).strip() if match.group(2) else ""
            text_content = match.group(3).strip()
            
            if title_hint:
                # Try to infer from explicit title
                speaker_role = infer_speaker_role(title_hint)
            else:
                speaker_role = infer_speaker_role(speaker_name)
            
            if text_content:
                segments.append(TranscriptSegment(
                    speaker_name=speaker_name,
                    speaker_role=speaker_role,
                    text=text_content,
                    line_number=line_number
                ))
    
    return ParsedTranscript(
        ticker=ticker,
        company_name=company_name,
        call_date=call_date,
        segments=segments,
        raw_text=text
    )


def filter_segments_by_role(parsed: ParsedTranscript, roles: List[SpeakerRole]) -> List[TranscriptSegment]:
    """Filter parsed transcript segments to include only specified roles."""
    return [seg for seg in parsed.segments if seg.speaker_role in roles]


def extract_management_statements(parsed: ParsedTranscript) -> str:
    """Extract concatenated text from CEO, CFO, COO speakers only."""
    mgmt_roles = [SpeakerRole.CEO, SpeakerRole.CFO, SpeakerRole.COO]
    mgmt_segments = filter_segments_by_role(parsed, mgmt_roles)
    return ' '.join(seg.text for seg in mgmt_segments)


def extract_analyst_questions(parsed: ParsedTranscript) -> str:
    """Extract concatenated text from Analyst speakers only."""
    analyst_segments = filter_segments_by_role(parsed, [SpeakerRole.ANALYST])
    return ' '.join(seg.text for seg in analyst_segments)


def segment_by_role(parsed: ParsedTranscript) -> dict:
    """Return dict mapping SpeakerRole to list of segments."""
    result = {}
    for role
