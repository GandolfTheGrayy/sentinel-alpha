"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for downstream LLM sentiment analysis in the Linguist pillar.
Detects speaker transitions via regex patterns, preserves speaker context, and returns
structured segments with confidence scores for role classification.

Used by linguist modules to perform role-specific tone and certainty analysis.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TranscriptSegment:
    """A single speaker segment from an earnings call transcript."""
    
    speaker_name: str
    speaker_role: str  # CEO, CFO, Analyst, Operator, Unknown
    text: str
    role_confidence: float  # 0.0 to 1.0
    start_line: int
    end_line: int


def parse_earnings_transcript(transcript_text: str) -> List[TranscriptSegment]:
    """
    Parse an earnings call transcript into speaker segments by role.
    
    Args:
        transcript_text: Raw transcript text, typically from earnings call platforms.
        
    Returns:
        List of TranscriptSegment objects, ordered by appearance.
    """
    lines = transcript_text.split('\n')
    segments: List[TranscriptSegment] = []
    
    current_speaker: Optional[str] = None
    current_role: str = "Unknown"
    current_role_confidence: float = 0.0
    current_text_lines: List[str] = []
    segment_start_line: int = 0
    
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect speaker transitions: patterns like "John Smith:" or "[CEO] John Smith"
        speaker_match = _detect_speaker_line(stripped)
        
        if speaker_match:
            # Save previous segment if it exists
            if current_speaker and current_text_lines:
                segment_text = '\n'.join(current_text_lines).strip()
                if segment_text:
                    segments.append(TranscriptSegment(
                        speaker_name=current_speaker,
                        speaker_role=current_role,
                        text=segment_text,
                        role_confidence=current_role_confidence,
                        start_line=segment_start_line,
                        end_line=line_idx - 1
                    ))
            
            # Parse new speaker
            current_speaker = speaker_match['name']
            current_role, current_role_confidence = _classify_speaker_role(
                speaker_match['name'],
                speaker_match.get('context', '')
            )
            current_text_lines = []
            segment_start_line = line_idx
        elif current_speaker and stripped:
            # Accumulate transcript text under current speaker
            current_text_lines.append(line)
        elif current_speaker and not stripped:
            # Preserve blank lines within a speaker's segment
            if current_text_lines:  # Only if we've started accumulating
                current_text_lines.append(line)
    
    # Flush final segment
    if current_speaker and current_text_lines:
        segment_text = '\n'.join(current_text_lines).strip()
        if segment_text:
            segments.append(TranscriptSegment(
                speaker_name=current_speaker,
                speaker_role=current_role,
                text=segment_text,
                role_confidence=current_role_confidence,
                start_line=segment_start_line,
                end_line=len(lines) - 1
            ))
    
    return segments


def _detect_speaker_line(line: str) -> Optional[dict]:
    """
    Detect if a line is a speaker label and extract name + context.
    
    Matches patterns:
      - "John Smith:" (name followed by colon)
      - "[CEO] John Smith" (role in brackets)
      - "John Smith (CEO)" (role in parentheses)
      
    Returns:
        Dict with 'name', 'context' keys, or None if no match.
    """
    if not line:
        return None
    
    # Pattern 1: "Name: " (most common)
    match = re.match(r'^([A-Z][A-Za-z\s\-\.]+?):\s*$', line)
    if match:
        return {'name': match.group(1).strip(), 'context': ''}
    
    # Pattern 2: "[ROLE] Name" or "[Role] Name"
    match = re.match(r'^\[([A-Za-z\s]+)\]\s+([A-Z][A-Za-z\s\-\.]+?)(?:\s|$)', line)
    if match:
        return {'name': match.group(2).strip(), 'context': match.group(1).strip()}
    
    # Pattern 3: "Name (ROLE)" or "Name (Role)"
    match = re.match(r'^([A-Z][A-Za-z\s\-\.]+?)\s*\(([A-Za-z\s]+?)\)\s*$', line)
    if match:
        return {'name': match.group(1).strip(), 'context': match.group(2).strip()}
    
    # Pattern 4: "Operator:" or "Moderator:" (single-word role + colon)
    match = re.match(r'^([A-Za-z]+):\s*$', line)
    if match:
        role_word = match.group(1)
        if role_word.lower() in ['operator', 'moderator', 'host', 'announcer']:
            return {'name': role_word, 'context': role_word}
    
    return None


def _classify_speaker_role(speaker_name: str, context: str) -> Tuple[str, float]:
    """
    Classify a speaker's role (CEO, CFO, Analyst, Operator) by name and context.
    
    Uses keyword matching on speaker name and context strings.
    Returns role name and confidence score (0.0–1.0).
    
    Args:
        speaker_name: Full name or role of the speaker.
        context: Additional context (e.g., from brackets or parentheses).
        
    Returns:
        Tuple of (role_string, confidence_float).
    """
    combined = (speaker_name + ' ' + context).lower()
    
    # High-confidence keywords
    if re.search(r'\bceo\b|\bchief\s+executive\b|\bpresident\b', combined):
        return ('CEO', 0.95)
    if re.search(r'\bcfo\b|\bchief\s+financial\b|\btreasurer\b', combined):
        return ('CFO', 0.95)
    if re.search(r'\bcoo\b|\bchief\s+operating\b', combined):
        return ('COO', 0.90)
    if re.search(r'\boperator\b|\bmodera?tor\b|\bhost\b', combined):
        return ('Operator', 0.98)
    
    # Medium-confidence keywords
    if re.search(r'\banalyst\b|\bquestion\b', combined):
        return ('Analyst', 0.70)
    if re.search(r'\bvp\b|\bvice\s+president\b|\bhead\s+of\b|\bsenior\b', combined):
        return ('Executive', 0.60)
    
    # Low confidence: generic speaker
    return ('Unknown', 0.0)


def segment_by_role(segments: List[TranscriptSegment], role: str) -> List[TranscriptSegment]:
    """
    Filter transcript segments to only those matching a given speaker role.
    
    Args:
        segments: List of all parsed segments.
        role: Role to filter by (e.g., "CEO", "Analyst").
        
    Returns:
        Filtered list of segments.
    """
    return [seg for seg in segments if seg.speaker_role.lower() == role.lower()]


def merge_adjacent_segments(segments: List[TranscriptSegment], role: str
