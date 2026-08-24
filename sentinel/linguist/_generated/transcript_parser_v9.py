"""
Earnings call transcript parser for Sentinel Sentiment Engine.

This module segments earnings call transcripts by speaker role (CEO, CFO, Analyst, etc.)
and prepares each segment for downstream LLM analysis via the Linguist pillar.
Enables role-aware sentiment and tone drift detection across executive vs. analyst perspectives.
"""

import re
from dataclasses import dataclass
from typing import list, dict, tuple


@dataclass
class TranscriptSegment:
    """A single speaker turn in a transcript with role and content."""
    role: str
    speaker_name: str
    text: str
    line_number: int


def detect_speaker_role(speaker_label: str) -> str:
    """Classify speaker role (CEO, CFO, Analyst, Operator, etc.) from label text."""
    label_lower = speaker_label.lower()
    
    role_keywords = {
        'ceo': ['chief executive', 'ceo', 'chief exec'],
        'cfo': ['chief financial', 'cfo', 'chief fin'],
        'coo': ['chief operating', 'coo', 'chief op'],
        'cto': ['chief technology', 'cto', 'chief tech'],
        'ir': ['investor relations', 'ir director', 'head of ir'],
        'analyst': ['analyst', 'research', 'managing director', 'md'],
        'operator': ['operator', 'moderator', 'host'],
    }
    
    for role, keywords in role_keywords.items():
        if any(kw in label_lower for kw in keywords):
            return role
    
    return 'unknown'


def parse_transcript(text: str) -> list[TranscriptSegment]:
    """
    Parse earnings call transcript into speaker segments with role classification.
    
    Handles common transcript formats:
    - "Speaker Name: text"
    - "Speaker Name (Title): text"
    - "Operator\ntext" (newline-separated)
    """
    segments: list[TranscriptSegment] = []
    lines = text.split('\n')
    
    current_speaker = None
    current_role = None
    current_text = []
    line_number = 0
    
    # Pattern: "Name (Title):" or "Name:" at line start
    speaker_pattern = re.compile(r'^([A-Z][A-Za-z\s\-\.]*?)\s*(?:\(([^)]+)\))?\s*:', re.MULTILINE)
    
    for i, line in enumerate(lines):
        line_number = i + 1
        line_stripped = line.strip()
        
        if not line_stripped:
            continue
        
        # Check if line starts a new speaker
        match = speaker_pattern.match(line)
        
        if match:
            # Save previous segment
            if current_speaker and current_text:
                combined_text = ' '.join(current_text).strip()
                if combined_text:
                    segments.append(TranscriptSegment(
                        role=current_role,
                        speaker_name=current_speaker,
                        text=combined_text,
                        line_number=line_number
                    ))
            
            # Start new segment
            current_speaker = match.group(1).strip()
            title_hint = match.group(2) if match.group(2) else current_speaker
            current_role = detect_speaker_role(title_hint)
            
            # Extract text after colon
            colon_pos = line.find(':')
            remainder = line[colon_pos + 1:].strip()
            current_text = [remainder] if remainder else []
        else:
            # Continuation of current speaker
            if current_speaker is not None:
                current_text.append(line_stripped)
    
    # Don't forget final segment
    if current_speaker and current_text:
        combined_text = ' '.join(current_text).strip()
        if combined_text:
            segments.append(TranscriptSegment(
                role=current_role,
                speaker_name=current_speaker,
                text=combined_text,
                line_number=line_number
            ))
    
    return segments


def group_by_role(segments: list[TranscriptSegment]) -> dict[str, list[TranscriptSegment]]:
    """Organize transcript segments by speaker role for role-aware analysis."""
    grouped: dict[str, list[TranscriptSegment]] = {}
    for segment in segments:
        if segment.role not in grouped:
            grouped[segment.role] = []
        grouped[segment.role].append(segment)
    return grouped


def extract_role_summary(segments: list[TranscriptSegment], role: str) -> str:
    """Concatenate all segments from a given role into a single analysis-ready string."""
    role_segments = [s for s in segments if s.role == role]
    texts = [s.text for s in role_segments]
    return ' '.join(texts)


def compute_role_statistics(segments: list[TranscriptSegment]) -> dict[str, dict]:
    """Calculate per-role metrics: speaker count, total words, average segment length."""
    grouped = group_by_role(segments)
    stats = {}
    
    for role, role_segs in grouped.items():
        word_counts = [len(s.text.split()) for s in role_segs]
        total_words = sum(word_counts)
        
        stats[role] = {
            'speaker_count': len(set(s.speaker_name for s in role_segs)),
            'segment_count': len(role_segs),
            'total_words': total_words,
            'avg_segment_length': total_words / len(role_segs) if role_segs else 0,
            'speakers': list(set(s.speaker_name for s in role_segs)),
        }
    
    return stats
