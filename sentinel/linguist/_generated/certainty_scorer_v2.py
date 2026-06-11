"""
Certainty vs. Hesitation scoring for corporate text via Claude LLM.

This module provides prompt templates and structured scoring for analyzing
corporate communications (earnings calls, SEC filings, earnings guidance) to
detect confidence signals ('certainty') versus cautious language ('hesitation').
Used by Linguist pillar to calibrate market prediction confidence.

Integrates with Claude Sonnet 4.6 for nuanced textual reasoning, returning
a CertaintyScore dataclass with numeric confidence (0.0-1.0), category label,
and supporting evidence snippets.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from anthropic import Anthropic


@dataclass
class CertaintyScore:
    """Structured output of linguistic certainty analysis."""
    
    ticker: str
    text_source: str  # e.g., "earnings_call", "8-K", "10-Q", "guidance"
    certainty_score: float  # 0.0 (highly hesitant) to 1.0 (highly certain)
    category: str  # "bullish_certain", "bullish_hesitant", "bearish_certain", "bearish_hesitant", "neutral"
    evidence_snippets: list[str] = field(default_factory=list)  # quotes supporting classification
    reasoning: str = ""  # brief explanation of score
    raw_response: str = ""  # full Claude response for audit trail


class CertaintyScorer:
    """LLM-powered analyzer of corporate linguistic confidence."""
    
    def __init__(self):
        """Initialize Anthropic client from ANTHROPIC_API_KEY env var."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY env var not set")
        self.client = Anthropic()
        self.model = "claude-sonnet-4-20250514"
    
    def score_text(
        self,
        ticker: str,
        text: str,
        source: str = "unknown"
    ) -> CertaintyScore:
        """
        Analyze corporate text for certainty vs. hesitation signals.
        
        Args:
            ticker: Stock ticker (e.g., "AAPL")
            text: Corporate text excerpt (earnings call, filing, guidance)
            source: Source label for audit (e.g., "earnings_call", "8-K")
        
        Returns:
            CertaintyScore with numeric confidence and supporting evidence.
        """
        prompt = self._build_scoring_prompt(ticker, text, source)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        raw_response = response.content[0].text
        score = self._parse_response(ticker, source, raw_response, text)
        score.raw_response = raw_response
        return score
    
    def _build_scoring_prompt(self, ticker: str, text: str, source: str) -> str:
        """Build the prompt template for certainty analysis."""
        return f"""You are a financial linguist analyzing corporate communications for confidence signals.

Ticker: {ticker}
Source: {source}

Corporate Text:
---
{text}
---

Analyze this text for:
1. CERTAINTY SIGNALS: Strong future tense, specific numbers, confident assertions ("will", "expects", "confident")
2. HESITATION SIGNALS: Qualifiers, disclaimers, uncertain phrasing ("may", "could", "if", "uncertain", "risks")
3. SENTIMENT DIRECTION: Is the overall tone bullish (growth, strength) or bearish (caution, headwinds)?

Output a JSON object with:
{{
  "certainty_score": <float 0.0-1.0, where 1.0=maximum certainty>,
  "category": <"bullish_certain"|"bullish_hesitant"|"bearish_certain"|"bearish_hesitant"|"neutral">,
  "evidence_snippets": [<list of 2-3 key quotes from text>],
  "reasoning": <2-3 sentence explanation>
}}

Be precise and cite specific phrases. Respond ONLY with the JSON object."""
    
    def _parse_response(
        self,
        ticker: str,
        source: str,
        response_text: str,
        original_text: str
    ) -> CertaintyScore:
        """Parse Claude's JSON response into a CertaintyScore."""
        import json
        
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            data = {
                "certainty_score": 0.5,
                "category": "neutral",
                "evidence_snippets": [],
                "reasoning": "Failed to parse LLM response"
            }
        
        return CertaintyScore(
            ticker=ticker,
            text_source=source,
            certainty_score=float(data.get("certainty_score", 0.5)),
            category=str(data.get("category", "neutral")),
            evidence_snippets=data.get("evidence_snippets", []),
            reasoning=data.get("reasoning", "")
        )


def score_earnings_call_excerpt(ticker: str, excerpt: str) -> CertaintyScore:
    """Quick helper: score an earnings call transcript excerpt."""
    scorer = CertaintyScorer()
    return scorer.score_text(ticker, excerpt, source="earnings_call")


def score_sec_filing(ticker: str, filing_text: str, form_type: str = "10-Q") -> CertaintyScore:
    """Quick helper: score an SEC filing (10-Q, 8-K, etc.)."""
    scorer = CertaintyScorer()
    return scorer.score_text(ticker, filing_text, source=form_type)


def score_guidance(ticker: str, guidance_text: str) -> CertaintyScore:
    """Quick helper: score forward guidance or outlook statements."""
    scorer = CertaintyScorer()
    return scorer.score_text(ticker, guidance_text, source="guidance")


if __name__ == "__main__":
    # Demo: score a mock earnings call excerpt
    sample_text = """
    We are very confident in our Q4 guidance of $50B revenue. We expect strong
    demand across all regions. However, there are risks around supply chain
    disruptions that could impact margins. We may see some headwinds in the
    China market due to regulatory uncertainty. Overall, we believe this will
    be a strong year for shareholder returns.
    """
    
    scorer = CertaintyScorer()
    result = scorer.score_text("AAPL", sample_text, source="earnings_call")
    
    print(f"Ticker: {result.ticker}")
    print(f"Source: {result.text_source}")
    print(f"Certainty Score: {result.certainty_score:.2f}")
    print(f"Category: {result.category}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Evidence: {result.evidence_snippets}")
