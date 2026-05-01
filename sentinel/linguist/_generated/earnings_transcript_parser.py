"""
Earnings call transcript parser for Sentinel Linguist pillar.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for downstream LLM analysis via the Linguist module.
Extracts speaker metadata, classifies role types, and normalizes text for
certainty/hesitation analysis and Linguistic Drift detection.

Integrates with sentinel/linguist/sample_score.py for tone analysis.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SpeakerRole(Enum):
    """Enumeration of recognized speaker roles in earnings calls."""
    CEO = "CEO"
    CFO = "CFO"
    OPERATOR = "Operator"
    ANALYST = "Analyst"
    OTHER = "Other"


@dataclass
class TranscriptSegment:
    """A single speaker's contiguous statement within an earnings transcript."""
    speaker_name: str
    speaker_role: SpeakerRole
    timestamp: Optional[str]
    text: str
    line_number: int

    def to_dict(self) -> dict:
        """Convert segment to dictionary for serialization."""
        return {
            "speaker_name": self.speaker_name,
            "speaker_role": self.speaker_role.value,
            "timestamp": self.timestamp,
            "text": self.text,
            "line_number": self.line_number,
        }


def infer_speaker_role(speaker_name: str, context_roles: dict[str, SpeakerRole]) -> SpeakerRole:
    """Infer speaker role from name patterns and accumulated context."""
    speaker_lower = speaker_name.lower().strip()

    # Explicit role patterns.
    if "operator" in speaker_lower or "moderator" in speaker_lower:
        return SpeakerRole.OPERATOR
    if "ceo" in speaker_lower or "chief executive" in speaker_lower:
        return SpeakerRole.CEO
    if "cfo" in speaker_lower or "chief financial" in speaker_lower:
        return SpeakerRole.CFO

    # If speaker appeared before, reuse cached role.
    if speaker_name in context_roles:
        return context_roles[speaker_name]

    # Heuristic: short names with punctuation often analysts; longer names execs.
    if len(speaker_name.split()) <= 2 and "," in speaker_name:
        return SpeakerRole.ANALYST

    return SpeakerRole.OTHER


def parse_earnings_transcript(text: str) -> list[TranscriptSegment]:
    """
    Parse a raw earnings call transcript into speaker segments.

    Recognizes common transcript formats:
    - "Speaker Name: [statement]"
    - "[HH:MM:SS] Speaker Name – [statement]"
    - "Operator: [statement]" for moderator interjections.

    Returns list of TranscriptSegment objects in order of appearance.
    """
    lines = text.split("\n")
    segments: list[TranscriptSegment] = []
    context_roles: dict[str, SpeakerRole] = {}
    current_speaker: Optional[str] = None
    current_role: Optional[SpeakerRole] = None
    current_text: list[str] = []
    current_timestamp: Optional[str] = None
    start_line: int = 0

    # Regex patterns for speaker headers.
    speaker_pattern = re.compile(
        r"^(?:\[?(\d{1,2}):(\d{2}):(\d{2})\]?)?\s*([A-Za-z\s,\.]+?)[\s]*(?::|–|-)\s*(.*)",
        re.MULTILINE
    )
    empty_line_pattern = re.compile(r"^\s*$")

    for line_no, line in enumerate(lines, start=1):
        # Check if line is a speaker header.
        match = speaker_pattern.match(line)

        if match:
            # Save previous segment if it exists.
            if current_speaker and current_text:
                combined_text = "\n".join(current_text).strip()
                if combined_text:
                    segments.append(
                        TranscriptSegment(
                            speaker_name=current_speaker,
                            speaker_role=current_role,
                            timestamp=current_timestamp,
                            text=combined_text,
                            line_number=start_line,
                        )
                    )

            # Parse new speaker header.
            hours = match.group(1)
            minutes = match.group(2)
            seconds = match.group(3)
            speaker_name = match.group(4).strip()
            initial_text = match.group(5).strip()

            # Build timestamp if present.
            if hours and minutes and seconds:
                current_timestamp = f"{hours}:{minutes}:{seconds}"
            else:
                current_timestamp = None

            current_speaker = speaker_name
            current_role = infer_speaker_role(speaker_name, context_roles)
            context_roles[speaker_name] = current_role
            current_text = [initial_text] if initial_text else []
            start_line = line_no

        elif empty_line_pattern.match(line):
            # Empty line might signal end of statement; continue accumulating.
            if current_text:
                current_text.append("")
        else:
            # Continuation of current speaker's statement.
            if current_speaker:
                current_text.append(line)

    # Save final segment.
    if current_speaker and current_text:
        combined_text = "\n".join(current_text).strip()
        if combined_text:
            segments.append(
                TranscriptSegment(
                    speaker_name=current_speaker,
                    speaker_role=current_role,
                    timestamp=current_timestamp,
                    text=combined_text,
                    line_number=start_line,
                )
            )

    return segments


def filter_segments_by_role(
    segments: list[TranscriptSegment], roles: list[SpeakerRole]
) -> list[TranscriptSegment]:
    """Filter transcript segments to only those matching specified roles."""
    return [seg for seg in segments if seg.speaker_role in roles]


def extract_executive_statements(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Extract statements from executives (CEO, CFO) only."""
    return filter_segments_by_role(segments, [SpeakerRole.CEO, SpeakerRole.CFO])


def extract_analyst_questions(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Extract statements from analysts and investors."""
    return filter_segments_by_role(segments, [SpeakerRole.ANALYST])


def normalize_segment_text(segment: TranscriptSegment) -> str:
    """Normalize transcript text: remove extra whitespace, standardize line breaks."""
    text = segment.text
    # Collapse multiple spaces.
    text = re.sub(r" {2,}", " ", text)
    # Collapse multiple newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def segment_to_analysis_prompt(segment: TranscriptSegment) -> str:
    """
    Format a transcript segment as a prompt for Linguist LLM analysis.

    Returns a formatted string ready for Claude to analyze tone/certainty.
    """
    role_str = segment.speaker_role.value
    normalized_text = normalize_segment_text(segment)

    prompt = (
        f"Speaker: {segment.speaker_name}\n"
        f"Role: {role_str}\n"
    )
    if segment.timestamp:
        prompt += f"Timestamp: {segment.timestamp}\n"

    prompt += f"\n{normalized_text}"
    return prompt


def merge_consecutive_speakers(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """
    Merge consecutive statements from the same speaker into a single segment.

    Useful for normalizing transcripts where speakers are fragmented across
    multiple header lines.
    """
    if not segments
