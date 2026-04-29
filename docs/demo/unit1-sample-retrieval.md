# Unit 1 Sample Retrieval Output

This acceptance artifact shows the normalized output shape for arXiv retrieval and local storage using the fixture in `tests/fixtures/arxiv_atom_response.xml`. Volatile `retrieved_at` timestamps are omitted from the sample for readability.

## Query

```json
{
  "topic": "agents",
  "category": "cs.LG",
  "start_date": "2026-04-18",
  "end_date": "2026-04-21",
  "start_index": 0,
  "max_results": 20
}
```

## arXiv Request Parameters

```json
{
  "search_query": "all:\"agents\" AND cat:cs.LG AND submittedDate:[202604180000 TO 202604212359]",
  "start": 0,
  "max_results": 20,
  "sortBy": "submittedDate",
  "sortOrder": "descending"
}
```

## Normalized Papers

```json
[
  {
    "paper_id": "2604.00001",
    "title": "Explainable Agents for Daily Research Briefings",
    "authors": ["Ada Lovelace", "Alan Turing"],
    "abstract": "We study agent workflows for explainable research-paper recommendation.",
    "categories": ["cs.LG", "cs.AI"],
    "published_date": "2026-04-20",
    "updated_date": "2026-04-20",
    "arxiv_url": "https://arxiv.org/abs/2604.00001",
    "pdf_url": "https://arxiv.org/pdf/2604.00001v1",
    "provenance": {
      "source": "arxiv",
      "source_url": "https://arxiv.org/abs/2604.00001v1",
      "query": "all:\"agents\" AND cat:cs.LG AND submittedDate:[202604180000 TO 202604212359]"
    }
  },
  {
    "paper_id": "2604.00002",
    "title": "Retrieval-Augmented Topic Tracking",
    "authors": ["Grace Hopper"],
    "abstract": "This paper presents retrieval methods for monitoring scientific topics.",
    "categories": ["cs.IR"],
    "published_date": "2026-04-18",
    "updated_date": "2026-04-19",
    "arxiv_url": "https://arxiv.org/abs/2604.00002",
    "pdf_url": "https://arxiv.org/pdf/2604.00002v2",
    "provenance": {
      "source": "arxiv",
      "source_url": "https://arxiv.org/abs/2604.00002v2",
      "query": "all:\"agents\" AND cat:cs.LG AND submittedDate:[202604180000 TO 202604212359]"
    }
  }
]
```

## Storage Behavior

- The retrieval run is stored under a stable key derived from the normalized query.
- A second retrieval with the same query returns the cached result set without calling arXiv again.
- Follow-up filtering can reuse stored metadata by topic, category, and date range.
- Normalized `paper_id` stays version-free, while `pdf_url` and `provenance.source_url` preserve the exact arXiv versioned source links used.
- PDF files are not downloaded in Unit 1.
