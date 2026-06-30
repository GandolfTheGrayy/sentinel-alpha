"""
Earnings call transcript parser for Sentinel Linguist.

Segments earnings transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares cleaned text segments for downstream LLM sentiment analysis.
Used by Linguist to detect tone shifts, confidence markers, and forward guidance
across different speaker personas within a single earnings call.

Typical flow:
  1. Raw transcript text → parse_transcript()
  2. Returns list of Segment objects (role, cleaned_text, timestamps)
  3. Each segment fed to LLM reasoning for certainty/hesitation markers
  4. RAG historian cross-references prior transcripts for drift detection
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """A single speaker's contiguous turn in an earnings call."""
    role: str
    text: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    line_number: int = 0


def parse_transcript(raw_text: str) -> list[Segment]:
    """
    Parse raw earnings transcript into role-segmented speaker turns.
    
    Handles common formats:
      - "Name, Title: text..." (Most common)
      - "Name (Title): text..."
      - "Operator: text..."
      - Timestamps like [00:12:34] or (00:12:34)
    
    Returns segments in order of appearance, with role normalized to one of:
    CEO, CFO, COO, ANALYST, OPERATOR, OTHER
    """
    lines = raw_text.split('\n')
    segments = []
    current_segment = None
    line_idx = 0
    
    # Regex patterns for speaker lines
    speaker_pattern = re.compile(
        r'^(?:\[[\d:]+\]\s*)?'  # optional timestamp prefix
        r'([^:(\n]+?)(?:\s*,\s*|\s*\()?'  # name
        r'([^):(\n]*(?:CEO|CFO|COO|President|Analyst|Operator|MD)[^):(\n]*)?'  # title
        r'[):]*\s*:\s*(.*)',
        re.IGNORECASE
    )
    
    timestamp_pattern = re.compile(r'\[?([\d:]+)\]?')
    
    for line_idx, line in enumerate(lines):
        if not line.strip():
            if current_segment and current_segment.text.strip():
                segments.append(current_segment)
                current_segment = None
            continue
        
        # Try to match speaker line
        match = speaker_pattern.match(line)
        if match:
            # Save prior segment
            if current_segment and current_segment.text.strip():
                segments.append(current_segment)
            
            name = match.group(1).strip()
            title = match.group(2).strip() if match.group(2) else ""
            first_text = match.group(3).strip() if match.group(3) else ""
            
            # Extract start timestamp if present
            ts_match = timestamp_pattern.search(line)
            start_time = ts_match.group(1) if ts_match else None
            
            # Normalize role from title or name
            role = _normalize_role(name, title)
            
            current_segment = Segment(
                role=role,
                text=first_text,
                start_time=start_time,
                line_number=line_idx
            )
        else:
            # Continuation of current speaker
            if current_segment:
                current_segment.text += " " + line.strip()
    
    # Don't forget last segment
    if current_segment and current_segment.text.strip():
        segments.append(current_segment)
    
    return segments


def _normalize_role(name: str, title: str) -> str:
    """Map speaker name/title to canonical role."""
    combined = (name + " " + title).upper()
    
    if "CEO" in combined:
        return "CEO"
    elif "CFO" in combined:
        return "CFO"
    elif "COO" in combined:
        return "COO"
    elif "OPERATOR" in combined or name.upper() == "OPERATOR":
        return "OPERATOR"
    elif "ANALYST" in combined or "QUESTION" in combined:
        return "ANALYST"
    else:
        return "OTHER"


def clean_segment_text(text: str) -> str:
    """
    Normalize segment text: remove filler words, standardize whitespace.
    
    Removes "um", "uh", "you know", etc. but preserves hesitation markers
    like "I think", "we believe" (important for certainty scoring).
    """
    # Remove common verbal fillers
    filler_pattern = re.compile(
        r'\b(um|uh|hmm|ah|err|erm)\b',
        re.IGNORECASE
    )
    text = filler_pattern.sub('', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def segment_by_topic_boundary(segments: list[Segment]) -> dict[str, list[Segment]]:
    """
    Group segments by detected topic (prepared remarks vs Q&A).
    
    Heuristic: "Operator:" or "Questions and Answers" marks transition from
    prepared remarks to Q&A. Useful for weighting CEO/CFO tone differently
    in prepared vs. questioned contexts.
    
    Returns dict: {"prepared_remarks": [...], "q_and_a": [...], "other": [...]}
    """
    prepared = []
    q_and_a = []
    other = []
    
    in_qa = False
    for seg in segments:
        if seg.role == "OPERATOR":
            in_qa = True
        
        if in_qa and seg.role in ("CEO", "CFO", "COO"):
            q_and_a.append(seg)
        elif not in_qa and seg.role in ("CEO", "CFO", "COO"):
            prepared.append(seg)
        else:
            other.append(seg)
    
    return {
        "prepared_remarks": prepared,
        "q_and_a": q_and_a,
        "other": other
    }


def extract_forward_guidance(segments: list[Segment]) -> list[str]:
    """
    Extract forward-looking statements from CEO/CFO segments.
    
    Looks for patterns like "expect", "guidance", "outlook", "forecast".
    Returns list of matched statements for downstream sentiment analysis.
    """
    guidance_pattern = re.compile(
        r'(?:we\s+)?(?:expect|guide|outlook|forecast|anticipate|project|believe|'
        r'target|plan|intend|estimate|see)\b[^.!?]*[.!?]',
        re.IGNORECASE
    )
    
    statements = []
    for seg in segments:
        if seg.role in ("CEO", "CFO", "COO"):
            matches = guidance_pattern.findall(seg.text)
            statements.extend(matches)
    
    return statements


def compute_speaker_statistics(segments: list[Segment]) -> dict[str, dict]:
    """
    Compute per-role word counts, turn length, and speaking time proxies.
    
    Returns dict mapping role → {"word_count", "turn_count", "avg_turn_length"}
    Useful for metadata filtering (e.g., weight CEO remarks more heavily).
    """
    stats = {}
    
    for role in ["CEO", "CFO", "COO", "ANALYST", "OPERATOR", "OTHER"]:
        role_segs = [s for s in segments if s.role == role]
        if not role_segs:
            continue
        
        total_words = sum(len(s.text.split()) for s in role_segs)
        turn_count = len(role_segs)
        avg_turn_length = total_words / turn_count if turn_count > 0 else 0
        
        stats[role] = {
            "word_count": total_words,
