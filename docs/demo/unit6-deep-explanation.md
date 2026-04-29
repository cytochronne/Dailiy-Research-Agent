# Unit 6 Deep Explanation Demo

This artifact shows the three selected-paper explanation modes from Unit 6 using:

- paper: `2604.00001` (`Explainable Agents for Daily Research Briefings`)
- source fixture: `tests/fixtures/sample_paper_text.txt`
- provider: `FakeLLMProvider`
- evidence source: `full_text`

## Method Mode

Summary: The paper studies an agent pipeline for daily arXiv recommendation and explanation.

Problem: The work addresses how to explain ranked research-paper recommendations for faster literature review.

Method overview: The paper proposes a multi-stage agent framework that retrieves papers, extracts structured evidence, and synthesizes selected-paper explanations.

Core workflow:
1. Retrieve candidate papers from arXiv
2. Rank them with topic and feedback signals
3. Generate targeted explanations from selected paper text

Inputs and outputs:
1. Inputs include a topic query, retrieved metadata, and selected paper text
2. Outputs include ranked recommendations and mode-specific explanations

Innovation: The claimed innovation is a transparent explanation loop that keeps evidence labels attached to every generated claim.

## Experiment Mode

Summary: The paper studies an agent pipeline for daily arXiv recommendation and explanation.

Datasets:
1. ArxivDailyBench
2. ResearchAgentEval

Baselines:
1. BM25 ranking
2. TF-IDF ranker
3. Dense retriever

Metrics:
1. Recall@5
2. MRR
3. preference win rate

Experimental setup: The authors evaluate the system on recent `cs.LG` papers and compare explanation quality with ablated variants.

Conclusions:
1. Full-text evidence improves explanation completeness
2. Abstract-only fallback remains useful when PDF parsing fails

## Limitations Mode

Summary: The paper studies an agent pipeline for daily arXiv recommendation and explanation.

Stated limitations:
1. The evaluation uses a small benchmark and only English arXiv papers

Assumptions:
1. The method assumes abstracts and PDFs are aligned with the final paper content

Missing validation:
1. The paper does not report latency under larger daily retrieval volumes

Risks:
1. Over-trusting generated explanations could hide missing experimental details from users

## Fallback Note

Automated coverage also verifies two non-demo fallback paths:

1. PDF parsing failure falls back to abstract-only explanation with a visible fallback status.
2. Missing experiment evidence in abstract-only mode is rendered as “not found in the available abstract source” instead of being invented.
