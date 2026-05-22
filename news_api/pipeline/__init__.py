"""Prediction-sentence ETL pipeline.

End-to-end flow:
1. Load cached NewsAPI CSVs from ``input_csvs/newsapi_cache/``.
2. Apply the excluded-domain detector (Tier 1 hosts, Tier 2 URL paths).
3. Clean article text (truncation marker, basic ad/boilerplate stripping).
4. Segment with ``pysbd`` and plan overlapping LLM windows.
5. Use Gemini for article-level domain assignment and sequential candidate
   span extraction (mockable for tests / dry-run).
6. Write a paired ``*_full-v*.json`` + ``*_predictions-v*.csv`` into
   ``annotators(1)/`` matching the existing annotator schema.

See ``ETL_Pipeline_Plan.md`` for the authoritative design (treat ``[x]``
decisions and the decision log as authoritative; ignore unchecked options
unless the user explicitly asks to revisit them).
"""

__all__ = [
    "config",
    "utils",
    "excluded_domains",
    "extract",
    "clean",
    "segment",
    "candidates",
    "llm_client",
    "transform",
    "validate",
    "state",
    "load",
    "run",
]
