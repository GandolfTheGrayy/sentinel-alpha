"""
Earnings Call Transcript Parser for Sentinel Sentiment Engine.

This module segments earnings call transcripts by speaker role (CEO, CFO, Analyst, etc.)
and prepares each segment for downstream LLM sentiment analysis in the Linguist pillar.
Parsed segments are tagged with speaker metadata, timestamps, and confidence scores,
enabling the Linguist to detect tone shifts, certainty levels, and regulatory whispers
across different company voices and Q&A phases.

Integration: Called by sentinel/linguist/sample_score.py during transcript ingestion;
output feeds directly into Claude-based certainty and drift analysis.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum


class SpeakerRole(Enum):
    """Enumeration of speaker roles in earnings calls."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    IR_OFFICER = "IR Officer"
    ANALYST = "Analyst"
    OPERATOR = "Operator"
    UNKNOWN = "Unknown"


@dataclass
class TranscriptSegment:
    """A contiguous utterance by one speaker in an earnings call."""
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    segment_index: int
    phase: str  # "prepared_remarks", "qa", "closing"
    timestamp_seconds: Optional[int] = None
    confidence: float = 1.0  # Speaker role detection confidence [0, 1]

    def to_dict(self) -> Dict:
        """Convert segment to dictionary for JSON serialization."""
        d = asdict(self)
        d["speaker_role"] = self.speaker_role.value
        return d


@dataclass
class ParsedTranscript:
    """Complete parsed earnings call transcript with metadata."""
    ticker: str
    date: str  # YYYY-MM-DD
    company_name: str
    segments: List[TranscriptSegment]
    raw_text: str
    metadata: Dict = None

    def __post_init__(self):
        """Initialize metadata dict if None."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert transcript to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "date": self.date,
            "company_name": self.company_name,
            "segments": [seg.to_dict() for seg in self.segments],
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }

    def segments_by_role(self, role: SpeakerRole) -> List[TranscriptSegment]:
        """Filter segments by speaker role."""
        return [seg for seg in self.segments if seg.speaker_role == role]

    def segments_by_phase(self, phase: str) -> List[TranscriptSegment]:
        """Filter segments by call phase (prepared_remarks, qa, closing)."""
        return [seg for seg in self.segments if seg.phase == phase]


def _detect_speaker_role(speaker_name: str, text_sample: str = "") -> Tuple[SpeakerRole, float]:
    """
    Infer speaker role from name and text context.
    
    Returns tuple of (role, confidence_score).
    Uses regex and keyword heuristics; high confidence on explicit titles.
    """
    name_lower = speaker_name.lower()
    text_lower = text_sample.lower()

    # Explicit role keywords in name or intro
    if re.search(r'\b(ceo|chief executive)\b', name_lower):
        return (SpeakerRole.CEO, 0.95)
    if re.search(r'\b(cfo|chief financial)\b', name_lower):
        return (SpeakerRole.CFO, 0.95)
    if re.search(r'\b(coo|chief operating)\b', name_lower):
        return (SpeakerRole.COO, 0.95)
    if re.search(r'\b(cto|chief technology)\b', name_lower):
        return (SpeakerRole.CTO, 0.95)
    if re.search(r'\b(ir\s|investor\s+relations|ir officer)\b', name_lower):
        return (SpeakerRole.IR_OFFICER, 0.90)
    if re.search(r'\b(operator|speaking|line)\b', text_lower):
        return (SpeakerRole.OPERATOR, 0.85)

    # Analyst indicators (question-heavy, external tone)
    if re.search(r'(analyst|fund|hedge|quant|managing director)', name_lower):
        return (SpeakerRole.ANALYST, 0.80)
    if re.search(r'^\s*(?:thank you|good morning|my question|can you)', text_lower):
        # Analyst questions often start formally
        if re.search(r'\?', text_sample):
            return (SpeakerRole.ANALYST, 0.70)

    # Soft heuristics on text content
    if re.search(r'(financial|guidance|margin|revenue|earnings)', text_lower):
        return (SpeakerRole.CFO, 0.60)
    if re.search(r'(operations|efficiency|execution|strategic)', text_lower):
        return (SpeakerRole.CEO, 0.55)

    return (SpeakerRole.UNKNOWN, 0.40)


def _split_into_phases(text: str) -> Dict[str, str]:
    """
    Heuristically split transcript into phases.
    
    Looks for common markers: "prepared remarks", "question and answer", "closing remarks".
    Returns dict mapping phase name to text chunk.
    """
    phases = {
        "prepared_remarks": "",
        "qa": "",
        "closing": "",
    }

    # Normalize text
    text_lower = text.lower()

    # Find phase boundaries
    qa_start = max(
        text_lower.find("question and answer"),
        text_lower.find("questions and answers"),
        text_lower.find("q&a"),
    )
    if qa_start == -1:
        qa_start = len(text)

    closing_start = max(
        text_lower.find("closing remarks"),
        text_lower.find("concluding remarks"),
        text_lower.find("thank you"),
    )
    if closing_start == -1:
        closing_start = len(text)

    # Ensure qa_start < closing_start for proper ordering
    if qa_start > closing_start:
        qa_start, closing_start = closing_start, qa_start

    phases["prepared_remarks"] = text[:qa_start]
    phases["qa"] = text[qa_start:closing_start]
    phases["closing"] = text[closing_start:]

    return phases


def _segment_by_speaker(text: str) -> List[Tuple[str, str]]:
    """
    Split text into (speaker_name, utterance) tuples.
    
    Handles common formats:
      - "Speaker Name: utterance text"
      - "Speaker Name\nutterance text"
      - Speaker tags like <Speaker>Name</Speaker> (if present)
    """
    segments = []

    # Pattern 1: "Name: text" or "Name\ntext"
    # Matches patterns like "John Doe: Lorem ipsum" or "John Doe\nLorem"
    pattern = r'^([A-Z][A-Za-z\s\.,-]+?):\s*(.+?)(?=\n[A-Z][A-Za-z\s\.,-]+?:|$)'
    matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)

    for match in matches:
        speaker_name = match.group(1).strip()
        utterance = match.group(2).strip()
        if speaker_name and utterance:
            segments.append((speaker_name, utterance))

    # If pattern 1 yielded nothing, try line-by-line fallback
    if not segments:
        current_speaker = None
