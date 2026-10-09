"""
Earnings call transcript parser for Sentinel Linguist.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for LLM sentiment analysis. Handles common transcript
formats (Motley Fool, Seeking Alpha, company IR websites) and normalizes speaker
attribution, timestamps, and role classification.

Output: list of TranscriptSegment objects, each tagged with speaker role, company
context, and raw text ready for certainty/drift analysis by Claude.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SpeakerRole(Enum):
    """Classification of speaker roles in earnings calls."""
    CEO = "ceo"
    CFO = "cfo"
    COO = "coo"
    CTO = "cto"
    PRESIDENT = "president"
    ANALYST = "analyst"
    INVESTOR = "investor"
    OPERATOR = "operator"
    OTHER = "other"


@dataclass
class TranscriptSegment:
    """
    A single speaker turn in an earnings call transcript.
    
    Attributes:
        speaker_name: Raw name as appearing in transcript.
        speaker_role: Classified role (CEO, CFO, Analyst, etc.).
        timestamp: Optional minute mark (e.g., "00:15:30").
        text: The spoken content.
        segment_idx: Position in transcript (0-indexed).
        company_ticker: Stock ticker for context (e.g., "AAPL").
        call_type: "earnings" | "shareholder_meeting" | "conference" | "unknown".
    """
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    segment_idx: int
    timestamp: Optional[str] = None
    company_ticker: Optional[str] = None
    call_type: str = "earnings"


def classify_speaker_role(name: str) -> SpeakerRole:
    """
    Infer speaker role from name string using regex patterns.
    
    Matches titles like "John Doe, Chief Financial Officer" or abbreviations
    like "Jane Smith, CFO". Falls back to analyst/other if not matched.
    """
    name_lower = name.lower()
    
    # CEO patterns
    if any(pat in name_lower for pat in ["chief executive officer", "ceo", " ceo "]):
        return SpeakerRole.CEO
    
    # CFO patterns
    if any(pat in name_lower for pat in ["chief financial officer", "cfo", " cfo "]):
        return SpeakerRole.CFO
    
    # COO patterns
    if any(pat in name_lower for pat in ["chief operating officer", "coo", " coo "]):
        return SpeakerRole.COO
    
    # CTO patterns
    if any(pat in name_lower for pat in ["chief technology officer", "cto", " cto "]):
        return SpeakerRole.CTO
    
    # President patterns
    if any(pat in name_lower for pat in ["president", "president and ceo"]):
        return SpeakerRole.PRESIDENT
    
    # Analyst/investor patterns
    if any(pat in name_lower for pat in ["analyst", "managing director", "portfolio manager", "fund manager"]):
        return SpeakerRole.ANALYST
    
    # Investor patterns
    if any(pat in name_lower for pat in ["investor", "shareholder"]):
        return SpeakerRole.INVESTOR
    
    # Operator patterns
    if any(pat in name_lower for pat in ["operator", "moderator"]):
        return SpeakerRole.OPERATOR
    
    return SpeakerRole.OTHER


def _extract_timestamp(line: str) -> Optional[str]:
    """
    Extract timestamp from a line if present (HH:MM:SS or MM:SS format).
    
    Returns ISO-like timestamp string or None.
    """
    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", line)
    if match:
        return f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
    match = re.search(r"(\d{1,2}):(\d{2})", line)
    if match:
        return f"00:{match.group(1)}:{match.group(2)}"
    return None


def parse_transcript(
    raw_text: str,
    company_ticker: Optional[str] = None,
    call_type: str = "earnings"
) -> list[TranscriptSegment]:
    """
    Parse earnings call transcript into speaker segments.
    
    Handles formats like:
      - "John Doe, CEO: [text]"
      - "John Doe\nCEO\n[text]"
      - "[timestamp] John Doe, CFO: [text]"
    
    Args:
        raw_text: Full transcript text (plaintext or HTML-stripped).
        company_ticker: Optional ticker for context (e.g., "AAPL").
        call_type: Type of call ("earnings", "shareholder_meeting", "conference").
    
    Returns:
        List of TranscriptSegment objects in order of appearance.
    """
    segments = []
    
    # Split on common speaker delimiters
    # Pattern: speaker name (optionally with title), then colon or newline, then text
    speaker_pattern = r"^(?:\[[\d:]+\])?\s*([A-Za-z\s.,\-\']+?)(?:\s*[:\—]|$)"
    
    lines = raw_text.split("\n")
    current_speaker = None
    current_text_lines = []
    current_timestamp = None
    segment_idx = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            continue
        
        # Check for speaker line
        match = re.match(speaker_pattern, line_stripped)
        if match:
            # Save previous segment if exists
            if current_speaker is not None and current_text_lines:
                segment_text = " ".join(current_text_lines).strip()
                if segment_text:
                    role = classify_speaker_role(current_speaker)
                    segments.append(
                        TranscriptSegment(
                            speaker_name=current_speaker,
                            speaker_role=role,
                            text=segment_text,
                            segment_idx=segment_idx,
                            timestamp=current_timestamp,
                            company_ticker=company_ticker,
                            call_type=call_type,
                        )
                    )
                    segment_idx += 1
            
            # Extract timestamp if present
            current_timestamp = _extract_timestamp(line_stripped)
            
            # Extract speaker name
            current_speaker = match.group(1).strip().rstrip(":")
            current_text_lines = []
            
            # If colon present, text may follow on same line
            colon_idx = line_stripped.find(":")
            if colon_idx != -1 and colon_idx < len(line_stripped) - 1:
                text_after_colon = line_stripped[colon_idx + 1:].strip()
                if text_after_colon:
                    current_text_lines.append(text_after_colon)
        else:
            # Continuation of current speaker's text
            if current_speaker is not None:
                current_text_lines.append(line_stripped)
    
    # Flush final segment
    if current_speaker is not None and current_text_lines:
        segment_text = " ".join(current_text_lines).strip()
        if segment_text:
            role = classify_speaker_role(current_speaker)
            segments.append(
                TranscriptSegment(
                    speaker_name=current_speaker,
                    speaker_role=role,
                    text=segment_text,
                    segment_idx=segment_idx,
                    timestamp=current_timestamp,
                    company_ticker=company_ticker,
                    call_type=call_type,
                )
            )
    
    return segments


def filter_by_role(
