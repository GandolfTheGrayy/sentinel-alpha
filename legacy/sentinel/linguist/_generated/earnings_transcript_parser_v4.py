"""
Earnings Call Transcript Parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, etc.)
and prepares each segment for downstream LLM sentiment analysis by the Linguist.
Detects speaker transitions, classifies role, and returns structured segments
with confidence metadata for use in tone-drift and certainty scoring.

This module is used by the Linguist pillar to isolate executive vs. analyst
sentiment signals before feeding them to Claude for nuanced reasoning.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class SpeakerRole(Enum):
    """Enumeration of recognized earnings call speaker roles."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    ANALYST = "Analyst"
    MODERATOR = "Moderator"
    OPERATOR = "Operator"
    UNKNOWN = "Unknown"


@dataclass
class TranscriptSegment:
    """A single speaker turn in an earnings call transcript."""
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    line_number: int
    role_confidence: float  # 0.0–1.0, how confident we are in the role classification


def _infer_role(speaker_name: str, context: str = "") -> Tuple[SpeakerRole, float]:
    """
    Infer speaker role from name and optional context.

    Returns (role, confidence) tuple.
    """
    name_lower = speaker_name.lower()
    context_lower = context.lower()

    # Keyword patterns for each role
    ceo_patterns = [r"\bceo\b", r"\bchief executive\b", r"\bpresident\b"]
    cfo_patterns = [r"\bcfo\b", r"\bchief financial\b", r"\btreasurer\b"]
    coo_patterns = [r"\bcoo\b", r"\bchief operating\b"]
    cto_patterns = [r"\bcto\b", r"\bchief technology\b", r"\bvp engineering\b"]
    analyst_patterns = [r"\banalyst\b", r"\banalysis\b"]
    moderator_patterns = [r"\bmoderator\b", r"\bfacilitator\b"]
    operator_patterns = [r"\boperator\b"]

    # Check name first
    for pattern in ceo_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.CEO, 0.95)
    for pattern in cfo_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.CFO, 0.95)
    for pattern in coo_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.COO, 0.90)
    for pattern in cto_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.CTO, 0.85)
    for pattern in moderator_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.MODERATOR, 0.90)
    for pattern in operator_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.OPERATOR, 0.95)
    for pattern in analyst_patterns:
        if re.search(pattern, name_lower):
            return (SpeakerRole.ANALYST, 0.80)

    # Check context (company name + role in parentheses, e.g., "John Doe, Company (Analyst)")
    if context:
        for pattern in ceo_patterns:
            if re.search(pattern, context_lower):
                return (SpeakerRole.CEO, 0.70)
        for pattern in cfo_patterns:
            if re.search(pattern, context_lower):
                return (SpeakerRole.CFO, 0.70)
        for pattern in analyst_patterns:
            if re.search(pattern, context_lower):
                return (SpeakerRole.ANALYST, 0.65)

    return (SpeakerRole.UNKNOWN, 0.0)


def parse_earnings_transcript(transcript_text: str) -> List[TranscriptSegment]:
    """
    Parse a raw earnings call transcript into speaker segments.

    Handles common formats:
    - "Speaker Name: text..."
    - "Speaker Name – text..."
    - "Speaker Name (Company, Title): text..."

    Returns a list of TranscriptSegment objects ordered by appearance.
    """
    segments: List[TranscriptSegment] = []
    lines = transcript_text.split("\n")
    
    # Pattern to match speaker intro (name at start of line, followed by colon or dash)
    # Optionally includes company/role info in parens
    speaker_pattern = r"^([A-Z][A-Za-z\s\-\.]+?)(?:\s*\(([^)]+)\))?\s*[:–—]\s*(.+)$"
    
    current_speaker: Optional[str] = None
    current_role: Optional[SpeakerRole] = None
    current_role_confidence: float = 0.0
    current_text: List[str] = []
    current_line_number: int = 0
    current_context: str = ""
    
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        
        if not stripped:
            continue
        
        match = re.match(speaker_pattern, stripped)
        
        if match:
            # Save previous segment if any
            if current_speaker and current_text:
                segment_text = "\n".join(current_text).strip()
                if segment_text:
                    segments.append(
                        TranscriptSegment(
                            speaker_name=current_speaker,
                            speaker_role=current_role or SpeakerRole.UNKNOWN,
                            text=segment_text,
                            line_number=current_line_number,
                            role_confidence=current_role_confidence,
                        )
                    )
            
            # Parse new speaker
            speaker_name = match.group(1).strip()
            context = match.group(2) or ""
            first_text = match.group(3).strip()
            
            current_speaker = speaker_name
            current_context = context
            current_role, current_role_confidence = _infer_role(speaker_name, context)
            current_text = [first_text] if first_text else []
            current_line_number = line_num
        else:
            # Continuation of current speaker's text
            if current_speaker:
                current_text.append(stripped)
    
    # Save final segment
    if current_speaker and current_text:
        segment_text = "\n".join(current_text).strip()
        if segment_text:
            segments.append(
                TranscriptSegment(
                    speaker_name=current_speaker,
                    speaker_role=current_role or SpeakerRole.UNKNOWN,
                    text=segment_text,
                    line_number=current_line_number,
                    role_confidence=current_role_confidence,
                )
            )
    
    return segments


def filter_segments_by_role(
    segments: List[TranscriptSegment],
    roles: List[SpeakerRole],
    min_confidence: float = 0.5,
) -> List[TranscriptSegment]:
    """
    Filter transcript segments to only those matching specified roles above confidence threshold.
    """
    return [
        seg for seg in segments
        if seg.speaker_role in roles and seg.role_confidence >= min_confidence
    ]


def extract_executive_sentiment_text(segments: List[TranscriptSegment]) -> str:
    """
    Extract and concatenate text from all C-suite executives for aggregated sentiment analysis.
    """
    executive_roles = {SpeakerRole.CEO, SpeakerRole.CFO, SpeakerRole.COO, SpeakerRole.C
