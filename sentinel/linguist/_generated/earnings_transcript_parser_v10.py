"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares normalized text segments for downstream LLM analysis. Integrates with
Linguist's certainty scoring and Historian's RAG pipeline by providing structured,
role-tagged utterances that preserve speaker identity and temporal sequence.

Used by Judge predictor to weight management guidance vs. analyst skepticism.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptSegment:
    """A single speaker turn in an earnings call."""
    speaker_name: str
    speaker_role: str  # "CEO", "CFO", "Analyst", "Operator", "Unknown"
    text: str
    sequence_idx: int  # Order in transcript (0-indexed)


@dataclass
class ParsedTranscript:
    """Complete parsed earnings call with metadata."""
    ticker: str
    date: str
    segments: list[TranscriptSegment]
    raw_text: str


def classify_speaker_role(speaker_name: str, context_text: str = "") -> str:
    """
    Infer speaker role from name and surrounding context.
    Returns one of: "CEO", "CFO", "Analyst", "Operator", "Unknown".
    """
    name_lower = speaker_name.lower()
    context_lower = context_text.lower()

    # Heuristic patterns for common titles embedded in names
    if any(x in name_lower for x in ["chief executive", "ceo", "president"]):
        return "CEO"
    if any(x in name_lower for x in ["chief financial", "cfo", "treasurer"]):
        return "CFO"
    if any(x in name_lower for x in ["operator", "moderator"]):
        return "Operator"

    # Check if recent context mentions analyst or analyst firm
    if "analyst" in context_lower or "question" in context_lower:
        return "Analyst"

    # Conservative default
    return "Unknown"


def parse_transcript(raw_transcript: str, ticker: str, date: str) -> ParsedTranscript:
    """
    Parse raw earnings call transcript into role-tagged segments.
    
    Expects format like:
      Operator
      Good morning and welcome...
      
      Jane Smith, Chief Executive Officer
      Thank you for joining us...
      
      Analyst: John Doe, Goldman Sachs
      What are your guidance assumptions?
    
    Returns ParsedTranscript with segments keyed by speaker role.
    """
    segments = []
    seq_idx = 0

    # Split by speaker lines (heuristic: line starting with name/title, optionally preceded by role)
    # Pattern: optional role on its own line, followed by name/title on next line
    lines = raw_transcript.split("\n")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this line looks like a speaker header
        # Patterns: "Name, Title", "Title: Name", "Operator", just "Name"
        if _is_speaker_header(line):
            speaker_name = _extract_speaker_name(line)
            speaker_role = classify_speaker_role(speaker_name, line)
            
            # Collect text until next speaker header
            text_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                if _is_speaker_header(next_line):
                    break
                text_lines.append(next_line)
                i += 1
            
            text = " ".join(text_lines).strip()
            
            if text:  # Only add if there's actual speech
                segment = TranscriptSegment(
                    speaker_name=speaker_name,
                    speaker_role=speaker_role,
                    text=text,
                    sequence_idx=seq_idx
                )
                segments.append(segment)
                seq_idx += 1
        else:
            i += 1
    
    return ParsedTranscript(
        ticker=ticker,
        date=date,
        segments=segments,
        raw_text=raw_transcript
    )


def _is_speaker_header(line: str) -> bool:
    """Heuristic: detect if a line is likely a speaker introduction."""
    # Very short lines (< 3 chars) are unlikely to be speech
    if len(line) < 3:
        return False
    
    # Lines that are entirely caps (like "OPERATOR") or contain titles/roles
    if line.isupper() or line == line.title():
        if any(keyword in line for keyword in [
            "Operator", "CEO", "CFO", "President", "Analyst",
            "Senior Vice President", "Vice President", "Manager",
            "Director", "Chief", "Officer"
        ]):
            return True
    
    # Lines with name, title pattern: "John Doe, Title" or "Title: John Doe"
    if "," in line or ": " in line:
        # Quick check: does it have a capitalized name-like structure?
        if re.search(r"[A-Z][a-z]+(\s+[A-Z][a-z]+)?", line):
            return True
    
    return False


def _extract_speaker_name(header_line: str) -> str:
    """Extract speaker name from a header line."""
    # Remove common prefixes
    text = header_line.strip()
    
    # Pattern: "Role: Name" or "Name, Role"
    if ": " in text:
        parts = text.split(": ", 1)
        if len(parts) == 2:
            # Return the part that looks more like a name (typically has fewer commas)
            candidate = parts[1] if "," not in parts[0] else parts[0]
            return candidate.split(",")[0].strip()
    
    if ", " in text:
        # "Name, Title" format
        return text.split(",")[0].strip()
    
    # Just a single name or role; return as-is
    return text


def segment_by_role(parsed: ParsedTranscript, role: str) -> list[TranscriptSegment]:
    """
    Filter parsed transcript segments by speaker role.
    Useful for analyzing only CEO/CFO vs. Analyst sentiment separately.
    """
    return [seg for seg in parsed.segments if seg.speaker_role == role]


def merge_consecutive_speakers(parsed: ParsedTranscript, max_gap: int = 2) -> list[TranscriptSegment]:
    """
    Merge consecutive turns by the same speaker, within a sequence gap threshold.
    Reduces fragmentation when speaker is interrupted by "Please continue" or operator remarks.
    """
    if not parsed.segments:
        return []
    
    merged = []
    current = parsed.segments[0]
    
    for i in range(1, len(parsed.segments)):
        seg = parsed.segments[i]
        # Merge if same role and sequence gap is small
        if (seg.speaker_role == current.speaker_role and 
            seg.sequence_idx - current.sequence_idx <= max_gap):
            current = TranscriptSegment(
                speaker_name=current.speaker_name,
                speaker_role=current.speaker_role,
                text=current.text + " " + seg.text,
                sequence_idx=current.sequence_idx
            )
        else:
            merged.append(current)
            current = seg
    
    merged.append(current)
    return merged


def normalize_text(text: str) -> str:
    """Clean and normalize transcript text for LLM analysis."""
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove common filler phrases (optional; tunable)
    text = re.sub(r"\b(um|uh|like|you know|so)\b", "", text, flags=re.IGNORECASE)
    return text
