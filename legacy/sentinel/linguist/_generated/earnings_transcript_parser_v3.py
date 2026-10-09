"""
Earnings call transcript parser for Sentinel Sentiment Engine.

Segments earnings call transcripts by speaker role (CEO, CFO, Analyst, etc.)
and extracts speaker-attributed statements for downstream LLM sentiment analysis.
Feeds parsed segments into Linguist's certainty scorer and Judge's prediction engine.

Typical usage:
  parser = EarningsTranscriptParser()
  segments = parser.parse(transcript_text)
  for segment in segments:
    print(f"{segment.speaker_role}: {segment.text[:100]}...")
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TranscriptSegment:
    """Single utterance within an earnings call, attributed to a speaker role."""
    speaker_name: str
    speaker_role: str  # "CEO", "CFO", "Analyst", "Operator", "Unknown"
    text: str
    line_number: int
    confidence: float  # 0.0–1.0; how confident the role classification is


class EarningsTranscriptParser:
    """Parse earnings call transcripts into role-attributed speaker segments."""

    # Common role indicators in earnings call headers.
    ROLE_PATTERNS = {
        "CEO": [
            r"(?:Chief Executive Officer|CEO)",
            r"(?:President|Founder)(?:\s+and\s+CEO)?",
        ],
        "CFO": [
            r"(?:Chief Financial Officer|CFO)",
            r"(?:Vice\s+President|VP).*(?:Finance|Financial)",
        ],
        "COO": [
            r"(?:Chief Operating Officer|COO)",
        ],
        "CTO": [
            r"(?:Chief Technology Officer|CTO)",
        ],
        "Analyst": [
            r"Analyst",
            r"(?:Goldman Sachs|Morgan Stanley|Citi|JP Morgan|Bank of America|Credit Suisse)",
        ],
        "Operator": [
            r"Operator",
        ],
    }

    # Regex to detect speaker lines: "Name Role" or "Name – Role" or "Name, Role".
    SPEAKER_LINE_PATTERN = re.compile(
        r"^([A-Z][a-z\s\-\.]+?)\s+(?:–|-|,|:)?\s*(.+?)$",
        re.MULTILINE,
    )

    # Heuristic: speaker labels appear at line starts, often indented.
    SPEAKER_PREFIX_PATTERN = re.compile(
        r"^\s*([A-Z][a-z\s\-\.]+?)\s*(?:–|-|:)\s*(.+?)$",
        re.MULTILINE,
    )

    def __init__(self) -> None:
        """Initialize the transcript parser with role detection patterns."""
        pass

    def classify_role(self, role_text: str) -> tuple[str, float]:
        """
        Classify a speaker's role from role descriptor text.

        Args:
            role_text: Raw role string (e.g., "Chief Financial Officer").

        Returns:
            Tuple of (role_category, confidence_0_to_1).
        """
        if not role_text:
            return "Unknown", 0.0

        role_text_lower = role_text.lower()

        # Try exact matches first (high confidence).
        for role, patterns in self.ROLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, role_text, re.IGNORECASE):
                    confidence = 0.95 if "Chief" in role_text or role in role_text_lower else 0.85
                    return role, confidence

        # Fallback: "Unknown" with low confidence.
        return "Unknown", 0.3

    def parse(self, transcript_text: str) -> List[TranscriptSegment]:
        """
        Parse an earnings call transcript into speaker-attributed segments.

        Args:
            transcript_text: Raw earnings call transcript text.

        Returns:
            List of TranscriptSegment objects, in order of appearance.
        """
        if not transcript_text or not transcript_text.strip():
            return []

        segments: List[TranscriptSegment] = []
        lines = transcript_text.split("\n")

        current_speaker_name: Optional[str] = None
        current_speaker_role: Optional[str] = None
        current_speaker_confidence: float = 0.0
        current_segment_lines: List[str] = []
        current_line_number: int = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Try to detect a speaker label line.
            speaker_match = self.SPEAKER_PREFIX_PATTERN.match(line)

            if speaker_match and len(stripped) < 150:
                # Looks like a speaker line. Save previous segment if any.
                if current_speaker_name and current_segment_lines:
                    segment_text = "\n".join(current_segment_lines).strip()
                    if segment_text:
                        segments.append(
                            TranscriptSegment(
                                speaker_name=current_speaker_name,
                                speaker_role=current_speaker_role or "Unknown",
                                text=segment_text,
                                line_number=current_line_number,
                                confidence=current_speaker_confidence,
                            )
                        )
                    current_segment_lines = []

                # Parse new speaker.
                name_part = speaker_match.group(1).strip()
                role_part = speaker_match.group(2).strip()

                current_speaker_name = name_part
                current_speaker_role, current_speaker_confidence = self.classify_role(role_part)
                current_line_number = i

                i += 1
                continue

            # Regular content line: append to current segment.
            if current_speaker_name:
                if stripped:  # Skip blank lines within segments.
                    current_segment_lines.append(line)

            i += 1

        # Don't forget the last segment.
        if current_speaker_name and current_segment_lines:
            segment_text = "\n".join(current_segment_lines).strip()
            if segment_text:
                segments.append(
                    TranscriptSegment(
                        speaker_name=current_speaker_name,
                        speaker_role=current_speaker_role or "Unknown",
                        text=segment_text,
                        line_number=current_line_number,
                        confidence=current_speaker_confidence,
                    )
                )

        return segments

    def filter_by_role(self, segments: List[TranscriptSegment], role: str) -> List[TranscriptSegment]:
        """
        Filter segments to only those matching a specific speaker role.

        Args:
            segments: List of transcript segments.
            role: Role to filter by (e.g., "CEO", "CFO", "Analyst").

        Returns:
            Filtered list of segments.
        """
        return [s for s in segments if s.speaker_role == role]

    def extract_executive_guidance(self, segments: List[TranscriptSegment]) -> str:
        """
        Concatenate segments from executives (CEO, CFO, COO, CTO).

        Args:
            segments: List of transcript segments.

        Returns:
            Combined text of all executive remarks.
        """
        exec_roles = {"CEO", "CFO", "COO", "CTO"}
        exec_segments = [s for s in segments if s.speaker_role in exec_roles]
        return "\n\n".join(s.text for s in exec_segments)

    def extract_analyst_questions(self, segments: List[TranscriptSegment]) -> str:
        """
        Concatenate segments from analysts (Q&A portion of call).

        Args:
            segments: List of transcript segments.

        Returns:
            Combined text of all analyst questions and remarks.
        """
        analyst_segments = self.filter_by_role(segments, "Analyst")
        return "\n\n".join(s.text for s in analyst_segments)


def parse_earnings_transcript(transcript_text: str) -> List[TranscriptSegment]:
    """
