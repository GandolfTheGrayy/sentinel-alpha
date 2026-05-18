"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to:
  1. Score tone shifts in company-specific communication over time
  2. Detect sentiment divergence between consecutive documents
  3. Flag anomalous linguistic patterns that precede market moves
  4. Assign confidence scores to drift signals

Tests use fixture text samples representing typical SEC filings, earnings calls,
and news articles to assert correct drift scoring logic.
"""

import pytest
from typing import Dict, List, Tuple


# ============================================================================
# Fixture Data: Representative Text Samples
# ============================================================================

@pytest.fixture
def sample_positive_filing() -> str:
    """Return boilerplate positive SEC filing text."""
    return """
    Form 10-Q — Quarterly Report
    
    Our financial performance exceeded expectations in Q3 2024. Revenue growth
    accelerated to 23% year-over-year, driven by strong demand in enterprise
    solutions. Operating margins expanded by 140 basis points, reflecting
    operational efficiency gains and favorable product mix.
    
    We remain confident in our market position and expect sustained momentum
    through year-end. Management is pleased with execution across all segments.
    """


@pytest.fixture
def sample_neutral_filing() -> str:
    """Return boilerplate neutral SEC filing text."""
    return """
    Form 10-Q — Quarterly Report
    
    Q3 2024 results were in line with guidance. Revenue reached $1.2B, with
    year-over-year growth of 12%. Operating expenses increased proportionally
    with headcount expansion in R&D.
    
    We continue to monitor macroeconomic conditions and competitive dynamics.
    The business remains stable with ongoing investments in next-gen products.
    """


@pytest.fixture
def sample_negative_filing() -> str:
    """Return boilerplate negative/cautious SEC filing text."""
    return """
    Form 10-Q — Quarterly Report
    
    Q3 2024 revenue declined 8% year-over-year to $980M, below prior guidance.
    Operating margins contracted due to elevated customer acquisition costs and
    supply chain headwinds. We face headwinds from intensifying competition.
    
    Management has initiated cost reduction initiatives. Near-term uncertainty
    persists in key verticals. We remain cautious about demand trends.
    """


@pytest.fixture
def sample_alarming_filing() -> str:
    """Return boilerplate highly negative/alarming SEC filing text."""
    return """
    Form 8-K — Current Report
    
    We are aware of material challenges impacting operations. Quarterly revenue
    fell 22% year-over-year. Customer churn accelerated unexpectedly. Several
    major contracts were terminated early.
    
    Management faces significant pressure to stabilize the business. Liquidity
    constraints may require strategic alternatives. Regulatory investigations
    have been initiated. Substantial doubt exists regarding continued operations.
    """


# ============================================================================
# Linguistic Drift Detector Stub (in-test implementation)
# ============================================================================

class LinguisticDriftDetector:
    """
    Minimal drift detector for testing.
    
    Scores documents on a sentiment scale (-1.0 to +1.0) and computes drift
    as the absolute change between consecutive documents.
    """

    @staticmethod
    def _score_document(text: str) -> float:
        """
        Score a document's sentiment on scale -1.0 (very negative) to +1.0 (very positive).
        
        Uses simple keyword matching as a proxy for full LLM reasoning.
        """
        positive_keywords = [
            "confidence", "momentum", "exceeded", "strong", "pleased",
            "growth", "accelerated", "expanded", "efficient", "leading"
        ]
        negative_keywords = [
            "declined", "headwinds", "challenges", "churn", "terminated",
            "cautious", "uncertainty", "pressure", "doubt", "alarming",
            "material", "constraints", "investigations"
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / max(total, 1)

    @staticmethod
    def detect_drift(documents: List[str]) -> List[Dict[str, float]]:
        """
        Compute drift scores between consecutive documents.
        
        Returns list of dicts with keys: 'index', 'score', 'drift', 'confidence'.
        """
        if len(documents) < 2:
            return []
        
        scores = [LinguisticDriftDetector._score_document(doc) for doc in documents]
        drifts = []
        
        for i in range(1, len(scores)):
            drift_magnitude = abs(scores[i] - scores[i - 1])
            confidence = 1.0 - (1.0 / (1.0 + drift_magnitude))  # Sigmoid-like
            
            drifts.append({
                'index': i,
                'score': scores[i],
                'drift': drift_magnitude,
                'confidence': confidence,
                'prev_score': scores[i - 1]
            })
        
        return drifts


# ============================================================================
# Test Cases
# ============================================================================

class TestLinguisticDriftDetector:
    """Test suite for Linguistic Drift detector."""

    def test_no_drift_identical_documents(self) -> None:
        """Assert zero drift when documents are identical."""
        text = "Revenue growth exceeded expectations. Strong momentum."
        drifts = LinguisticDriftDetector.detect_drift([text, text])
        
        assert len(drifts) == 1
        assert drifts[0]['drift'] < 0.01, "Identical docs should have near-zero drift"

    def test_positive_to_neutral_drift(
        self, sample_positive_filing: str, sample_neutral_filing: str
    ) -> None:
        """Assert detectable drift from positive to neutral tone."""
        drifts = LinguisticDriftDetector.detect_drift([
            sample_positive_filing,
            sample_neutral_filing
        ])
        
        assert len(drifts) == 1
        assert drifts[0]['drift'] > 0.1, "Positive→Neutral should show measurable drift"
        assert drifts[0]['score'] < drifts[0]['prev_score'], "Score should decrease"

    def test_neutral_to_negative_drift(
        self, sample_neutral_filing: str, sample_negative_filing: str
    ) -> None:
        """Assert detectable drift from neutral to negative tone."""
        drifts = LinguisticDriftDetector.detect_drift([
            sample_neutral_filing,
            sample_negative_filing
        ])
        
        assert len(drifts) == 1
        assert drifts[0]['drift'] > 0.15, "Neutral→Negative should show stronger drift"
        assert drifts[0]['score'] < 0, "Negative filing should have negative score"

    def test_cumulative_drift_sequence(
        self,
        sample_positive_filing: str,
        sample_neutral_filing: str,
        sample_negative_filing: str,
        sample_alarming_filing: str
    ) -> None:
        """Assert escalating drift across multi-document sequence."""
        documents = [
            sample_positive_filing,
            sample_neutral_filing,
            sample_negative_filing,
            sample_alarming_filing
        ]
        drifts = LinguisticDriftDetector.detect_drift(documents)
        
        assert len(drifts) == 3, "Should produce 3 drift measurements"
        
        # Each successive drift should be substantial
        for i, drift_entry in enumerate(drifts):
            assert drift_entry['drift'] > 0.1, f"Drift {i} should exceed threshold"
        
        # Drift should generally increase or stabilize (worsening tone)
        assert drifts[-1]['score'] < drifts[0]['score'], \
