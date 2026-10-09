"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, Operator)
and prepares each segment for LLM sentiment analysis. Handles common transcript
formats (seeking alpha, investor.com, company IR portals) and extracts speaker
identity, role classification, and normalized text.

This module feeds the Linguist pillar's certainty and tone-shift analysis by
providing clean, role-tagged segments ready for Claude reasoning.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SpeakerRole(Enum):
    """Enumeration of recognized speaker roles in earnings calls."""
    CEO = "CEO"
    CFO = "CFO"
    CTO = "CTO"
    COO = "COO"
    ANALYST = "Analyst"
    OPERATOR = "Operator"
    OTHER_EXEC = "Other Executive"
    UNKNOWN = "Unknown"


@dataclass
class TranscriptSegment:
    """A single speaker turn in an earnings call transcript."""
    speaker_name: str
    speaker_role: SpeakerRole
    text: str
    timestamp_seconds: Optional[int] = None
    line_number: int = 0

    def __repr__(self) -> str:
        """Return a readable representation of this segment."""
        role_str = self.speaker_role.value
        ts_str = f" @ {self.timestamp_seconds}s" if self.timestamp_seconds else ""
        return f"[{role_str}: {self.speaker_name}{ts_str}] {self.text[:60]}..."


class TranscriptParser:
    """
    Parse earnings call transcripts into role-tagged segments.
    
    Supports multiple transcript formats and normalizes speaker names and roles.
    """

    # Common role keywords for classification
    EXEC_TITLES = {
        "ceo": SpeakerRole.CEO,
        "chief executive": SpeakerRole.CEO,
        "cfo": SpeakerRole.CFO,
        "chief financial": SpeakerRole.CFO,
        "treasurer": SpeakerRole.CFO,
        "cto": SpeakerRole.CTO,
        "chief technology": SpeakerRole.CTO,
        "coo": SpeakerRole.COO,
        "chief operating": SpeakerRole.COO,
        "president": SpeakerRole.OTHER_EXEC,
        "vp": SpeakerRole.OTHER_EXEC,
        "vice president": SpeakerRole.OTHER_EXEC,
        "chairman": SpeakerRole.OTHER_EXEC,
        "board member": SpeakerRole.OTHER_EXEC,
    }

    OPERATOR_KEYWORDS = {"operator", "moderator", "call coordinator"}
    ANALYST_KEYWORDS = {"analyst", "fund manager", "investor", "portfolio manager"}

    def __init__(self) -> None:
        """Initialize the transcript parser."""
        pass

    def parse(self, text: str) -> list[TranscriptSegment]:
        """
        Parse a transcript string into segments by speaker.
        
        Detects speaker labels (e.g., "John Smith, CEO:" or "Analyst:") and
        segments text accordingly. Returns a list of TranscriptSegment objects.
        """
        segments = []
        lines = text.split("\n")
        current_speaker = None
        current_role = SpeakerRole.UNKNOWN
        current_text = []
        line_num = 0

        # Pattern: "Name, Title:" or "Name:" or "[Title] Name:" or just "Title:"
        speaker_pattern = re.compile(
            r"^(?:\[)?([A-Za-z\s\.\,\-\']+?)(?:\])?(?:\s*,\s*)?([A-Za-z\s\-]*?):\s*(.*)$",
            re.MULTILINE,
        )

        for line_num, line in enumerate(lines, start=1):
            line = line.rstrip()
            if not line:
                continue

            # Check if line starts with a speaker label
            match = speaker_pattern.match(line)
            if match:
                # Save previous segment if exists
                if current_speaker and current_text:
                    text_content = " ".join(current_text).strip()
                    if text_content:
                        segments.append(
                            TranscriptSegment(
                                speaker_name=current_speaker,
                                speaker_role=current_role,
                                text=text_content,
                                line_number=line_num,
                            )
                        )

                # Parse new speaker
                name_raw = match.group(1).strip()
                title_raw = match.group(2).strip()
                first_words = match.group(3).strip()

                current_speaker = name_raw or title_raw or "Unknown"
                current_role = self._classify_role(name_raw, title_raw)
                current_text = [first_words] if first_words else []
            else:
                # Continuation of current speaker's text
                if line.strip():
                    current_text.append(line.strip())

        # Save final segment
        if current_speaker and current_text:
            text_content = " ".join(current_text).strip()
            if text_content:
                segments.append(
                    TranscriptSegment(
                        speaker_name=current_speaker,
                        speaker_role=current_role,
                        text=text_content,
                        line_number=line_num,
                    )
                )

        return segments

    def _classify_role(self, name: str, title: str) -> SpeakerRole:
        """
        Classify a speaker's role based on name and title fields.
        
        Returns the most likely SpeakerRole enum value.
        """
        combined = f"{name} {title}".lower()

        # Check for operator/moderator
        if any(kw in combined for kw in self.OPERATOR_KEYWORDS):
            return SpeakerRole.OPERATOR

        # Check for analyst/investor
        if any(kw in combined for kw in self.ANALYST_KEYWORDS):
            return SpeakerRole.ANALYST

        # Check for executive titles
        for keyword, role in self.EXEC_TITLES.items():
            if keyword in combined:
                return role

        # Default: if it's in the "questions and answers" section, assume analyst
        # (This heuristic can be refined per-company later)
        return SpeakerRole.UNKNOWN

    def segment_by_role(
        self, segments: list[TranscriptSegment], role: SpeakerRole
    ) -> list[TranscriptSegment]:
        """
        Filter segments to only those from a specific speaker role.
        
        Useful for isolating management vs. analyst sentiment.
        """
        return [s for s in segments if s.speaker_role == role]

    def segment_by_name(
        self, segments: list[TranscriptSegment], name: str
    ) -> list[TranscriptSegment]:
        """
        Filter segments to only those from a specific speaker name.
        """
        name_lower = name.lower()
        return [s for s in segments if s.speaker_name.lower() == name_lower]

    def get_role_summary(self, segments: list[TranscriptSegment]) -> dict[str, int]:
        """
        Count segments by speaker role.
        
        Returns a dict mapping role names to segment counts.
        """
        summary = {}
        for segment in segments:
            role_key = segment.speaker_role.value
            summary[role_key] = summary.get(role_key, 0) + 1
        return summary

    def get_speaker_summary(self, segments: list[TranscriptSegment]) -> dict[str, int]:
        """
        Count segments by individual speaker name.
        
        Returns a dict mapping speaker names to segment counts.
        """
        summary = {}
        for segment in segments:
            summary[segment.speaker_name] = summary.get(segment.speaker_name, 0) + 1
        return summary


def parse_transcript(transcript_text: str) ->
