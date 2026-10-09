"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for LLM linguistic analysis. Detects speaker transitions,
normalizes timestamps, and tags segments with confidence scores based on role clarity.

Integrated into the Linguist pillar to support tone drift detection and
certainty/hesitation analysis across executive vs. analyst commentary.
"""

import re
from dataclasses import dataclass
from typing import list, dict, tuple, Optional


@dataclass
class Segment:
    """A parsed segment of an earnings call transcript."""
    role: str
    speaker_name: str
    timestamp: Optional[str]
    text: str
    confidence: float


def normalize_speaker_role(role_str: str) -> str:
    """Normalize speaker role string to canonical form."""
    role_lower = role_str.lower().strip()
    if any(x in role_lower for x in ["ceo", "chief executive"]):
        return "CEO"
    if any(x in role_lower for x in ["cfo", "chief financial"]):
        return "CFO"
    if any(x in role_lower for x in ["coo", "chief operating"]):
        return "COO"
    if any(x in role_lower for x in ["president"]):
        return "President"
    if any(x in role_lower for x in ["analyst", "question"]):
        return "Analyst"
    if any(x in role_lower for x in ["operator"]):
        return "Operator"
    return "Unknown"


def extract_timestamp(line: str) -> Optional[str]:
    """Extract timestamp from a line if present (HH:MM:SS format)."""
    match = re.search(r"\d{1,2}:\d{2}:\d{2}", line)
    return match.group(0) if match else None


def detect_speaker_transition(line: str) -> Optional[tuple[str, str]]:
    """
    Detect speaker name and role from a transition line.
    
    Returns (speaker_name, role_string) or None if no transition detected.
    Handles patterns like "John Smith, CEO" or "[CEO] Jane Doe" or "Operator:"
    """
    line = line.strip()
    
    # Pattern 1: "Name, Title" or "Name – Title"
    match = re.match(r"^([A-Za-z\s\.]+)[,–]\s*([A-Za-z\s]+?)(?:\s*$|\s*–)", line)
    if match:
        return (match.group(1).strip(), match.group(2).strip())
    
    # Pattern 2: "[TITLE] Name"
    match = re.match(r"^\[([A-Z\s]+)\]\s+([A-Za-z\s\.]+?)(?:\s*$)", line)
    if match:
        return (match.group(2).strip(), match.group(1).strip())
    
    # Pattern 3: "Name (Title):"
    match = re.match(r"^([A-Za-z\s\.]+?)\s*\(([A-Za-z\s]+?)\)\s*:", line)
    if match:
        return (match.group(1).strip(), match.group(2).strip())
    
    # Pattern 4: "Operator:" or "CEO:" etc. at line start
    match = re.match(r"^([A-Za-z]+?):\s*$", line)
    if match:
        role_candidate = match.group(1).strip()
        if any(x in role_candidate.lower() for x in ["ceo", "cfo", "operator", "analyst"]):
            return (role_candidate, role_candidate)
    
    return None


def parse_transcript(text: str, min_segment_length: int = 20) -> list[Segment]:
    """
    Parse an earnings call transcript into speaker segments.
    
    Args:
        text: Full transcript text.
        min_segment_length: Minimum character length to retain a segment.
    
    Returns:
        List of Segment objects, sorted by appearance.
    """
    lines = text.split("\n")
    segments = []
    
    current_speaker = None
    current_role = None
    current_text = []
    current_timestamp = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Try to detect a speaker transition
        transition = detect_speaker_transition(line_stripped)
        
        if transition:
            speaker_name, role_str = transition
            
            # Flush accumulated text from previous speaker
            if current_speaker and current_text:
                combined_text = " ".join(current_text).strip()
                if len(combined_text) >= min_segment_length:
                    role_normalized = normalize_speaker_role(current_role or "Unknown")
                    confidence = _compute_role_confidence(current_role, role_normalized)
                    segments.append(
                        Segment(
                            role=role_normalized,
                            speaker_name=current_speaker,
                            timestamp=current_timestamp,
                            text=combined_text,
                            confidence=confidence
                        )
                    )
            
            # Start new speaker
            current_speaker = speaker_name
            current_role = role_str
            current_text = []
            current_timestamp = extract_timestamp(line_stripped)
        else:
            # Accumulate text for current speaker
            if line_stripped and current_speaker:
                current_text.append(line_stripped)
            elif line_stripped and not current_speaker:
                # Orphan line before first speaker detected; skip or assign to "Unknown"
                pass
    
    # Flush final speaker
    if current_speaker and current_text:
        combined_text = " ".join(current_text).strip()
        if len(combined_text) >= min_segment_length:
            role_normalized = normalize_speaker_role(current_role or "Unknown")
            confidence = _compute_role_confidence(current_role, role_normalized)
            segments.append(
                Segment(
                    role=role_normalized,
                    speaker_name=current_speaker,
                    timestamp=current_timestamp,
                    text=combined_text,
                    confidence=confidence
                )
            )
    
    return segments


def filter_by_role(segments: list[Segment], roles: list[str]) -> list[Segment]:
    """Filter segments to only those matching given roles."""
    return [s for s in segments if s.role in roles]


def summarize_segments(segments: list[Segment]) -> dict[str, dict]:
    """
    Generate summary statistics by role.
    
    Returns dict mapping role -> {count, total_chars, avg_confidence}
    """
    role_stats = {}
    for seg in segments:
        if seg.role not in role_stats:
            role_stats[seg.role] = {
                "count": 0,
                "total_chars": 0,
                "confidence_sum": 0.0
            }
        role_stats[seg.role]["count"] += 1
        role_stats[seg.role]["total_chars"] += len(seg.text)
        role_stats[seg.role]["confidence_sum"] += seg.confidence
    
    # Compute averages
    for role in role_stats:
        count = role_stats[role]["count"]
        if count > 0:
            role_stats[role]["avg_confidence"] = role_stats[role]["confidence_sum"] / count
        else:
            role_stats[role]["avg_confidence"] = 0.0
        del role_stats[role]["confidence_sum"]
    
    return role_stats


def _compute_role_confidence(raw_role: Optional[str], normalized_role: str) -> float:
    """
    Compute confidence score (0–1) for role identification.
    
    Higher confidence if role string matched canonical patterns closely.
    """
    if not raw_role or normalized_role == "Unknown":
        return 0.5
    
    raw_lower = raw_role.lower()
    
    # Exact substring matches get
