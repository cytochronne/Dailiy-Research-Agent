# Daily arXiv Research Briefing Agent

Local Agent + Skills project for daily arXiv paper monitoring, recommendation, feedback refinement, briefing generation, and selected-paper explanation.

## Current Status

This repository is being built in staged units from `docs/plans/2026-04-21-001-feat-daily-arxiv-agent-plan.md`.

Current stage: Unit 3, seed-paper personalization MVP.

## Setup

Use the dedicated conda environment for all Python work:

```bash
conda activate daily-arxiv-agent
```

If the environment does not exist yet, create it with the Unit 0 dependencies:

```bash
conda env create -f environment.yml
conda activate daily-arxiv-agent
python -m pip install -e . --no-build-isolation --no-deps
```

All future Python packages should be installed into this conda environment. Prefer conda packages when available; use pip only inside the active conda environment when a package is not available through conda.

Copy `.env.example` to `.env` for local settings. Keep real secrets out of git.

## Test

```bash
conda run -n daily-arxiv-agent python -m pytest
```

## arXiv Retrieval and Local Storage

Unit 1 adds an independently testable retrieval Skill and SQLite store:

- `daily_arxiv_agent.skills.arxiv_retrieval.ArxivRetrievalSkill`
- `daily_arxiv_agent.storage.SQLitePaperStore`
- `daily_arxiv_agent.contracts.RetrievalQuery`

The retrieval Skill supports topic, category, submitted-date range, pagination, Atom parsing, local caching, and structured fallback results. It stores normalized metadata only; PDFs are not downloaded in this unit.

Example:

```python
from datetime import date

from daily_arxiv_agent.contracts import RetrievalQuery
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.storage import SQLitePaperStore

store = SQLitePaperStore("data/daily_arxiv.sqlite3")
skill = ArxivRetrievalSkill(store=store)
result = skill.retrieve(
    RetrievalQuery(
        topic="agent briefing",
        category="cs.LG",
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 21),
        max_results=10,
    )
)

papers = result.data or []
```

Follow-up filtering can reuse stored papers without a new arXiv request:

```python
stored = store.find_papers(
    topic="briefing",
    category="cs.LG",
    start_date=date(2026, 4, 18),
    end_date=date(2026, 4, 21),
)
```

## Topic Ranking and Daily Briefing

Unit 2 adds deterministic keyword ranking, structured paper extraction behind an LLM provider boundary, and first-pass daily briefing generation:

- `daily_arxiv_agent.skills.ranking.TopicRankingSkill`
- `daily_arxiv_agent.skills.extraction.PaperExtractionSkill`
- `daily_arxiv_agent.skills.briefing.DailyBriefingSkill`
- `daily_arxiv_agent.llm.fake.FakeLLMProvider`

The default provider is `fake`, so ranking, extraction, briefing generation, and tests run without live LLM credentials. Every briefing item carries an evidence label (`metadata` or `abstract`) and preserves the source arXiv provenance.

Example:

```python
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.briefing import DailyBriefingSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill

ranking = TopicRankingSkill().rank(
    papers,
    topic="agent briefing",
    top_k=5,
)
recommendations = ranking.data or []

briefing = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
    topic="agent briefing",
    recommendations=recommendations,
)

daily_briefing = briefing.data
```

## Seed-Paper Personalization

Unit 3 adds seed parsing, deterministic preference vectors, seed-aware ranking, and SQLite persistence for local reuse:

- `daily_arxiv_agent.skills.seed_parsing.SeedParsingSkill`
- `daily_arxiv_agent.skills.seed_parsing.DeterministicTextVectorizer`
- `daily_arxiv_agent.contracts.SeedPreference`
- `SQLitePaperStore.save_seed_preference(...)`
- `SQLitePaperStore.load_seed_preference(...)`

Seed inputs support arXiv IDs, arXiv URLs, and title text. arXiv ID/URL seeds attempt metadata resolution when a metadata client is available; title-only seeds still contribute preference text and do not require API success.

Example:

```python
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill
from daily_arxiv_agent.storage import SQLitePaperStore

seed_result = SeedParsingSkill().build_preference(
    [
        "2604.00001",
        "https://arxiv.org/abs/2604.00002v1",
        "Agent workflows for research paper recommendation",
    ],
    profile_id="default",
)
seed_preference = seed_result.data

store = SQLitePaperStore("data/daily_arxiv.sqlite3")
if seed_preference is not None:
    store.save_seed_preference(seed_preference)

stored_preference = store.load_seed_preference("default")
ranking = TopicRankingSkill().rank(
    papers,
    seed_preference=stored_preference,
    top_k=5,
)
recommendations = ranking.data or []
```

Seed ranking can also be combined with explicit topic ranking:

```python
hybrid = TopicRankingSkill().rank(
    papers,
    topic="agent briefing",
    seed_preference=stored_preference,
    top_k=5,
)
```

## Planned Demo

The final local demo will show:

- Agent workflow trace
- arXiv retrieval results
- ranked recommendations
- daily briefing
- like/dislike refinement
- selected-paper deep explanation modes

## Staged Delivery Rule

Each implementation unit is developed and verified independently. After a unit is implemented:

1. Run the unit's automated checks.
2. Produce its manual acceptance artifact under `docs/demo/`.
3. Ask for user acceptance.
4. Commit and push only after acceptance.
5. Start the next unit only after the pushed commit is confirmed.

See `docs/demo/staged-acceptance.md` for the full acceptance checklist.
