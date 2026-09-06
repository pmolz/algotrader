"""External research briefs — the idea supply the local model codes from.

The nightly loop's proposer is a 7B coder model. It is good at turning a
specified idea into contract-correct Python and bad at inventing ideas: left to
itself it converges on the same handful of textbook mechanisms, which is what
`memory.family_tally` exists to push back against. A brief is the other half of
that fix. Something with a wider view of the literature (Claude, on a schedule —
see `docs/RESEARCH.md`) reads around, discards what this framework cannot test,
and writes the survivors down in the shape the propose prompt already wants.

The division of labour is the whole point:

  brief  = the idea, the mechanism, the ranked score, the starting parameters
  7B     = naming, the class contract, the hold pattern, the risk overlay

A brief written as an essay does not survive contact with a 7B — the same reason
`prompts.WORKED_EXAMPLE` exists. So briefs are short, sectioned, and capped at
`render()`'s character budget, because they are competing for attention with the
lessons block and the worked example in the same prompt.

Nothing here decides whether an idea is any good. A brief buys the candidate a
place in the queue and nothing else; the gauntlet is unchanged and does not know
where a candidate came from. `experiments.researched_from` records the brief id
so that afterwards you can ask whether researched ideas actually cleared more
stages than self-generated ones — a question worth being able to answer, given
that published edges are the ones most likely to be already arbitraged away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# A brief is YAML front matter (metadata the loader needs) followed by the body
# that is pasted into the prompt verbatim. Keeping the body free text rather
# than more YAML is deliberate: it is prompt copy, and it should be edited and
# read as prompt copy.
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)

_REQUIRED = ("id", "title", "family")
_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{2,63}\Z")


class BriefError(ValueError):
    """A brief file that cannot be trusted to be what it claims."""


@dataclass(frozen=True)
class Brief:
    id: str
    title: str
    family: str
    body: str
    sources: tuple[str, ...] = ()
    retired: bool = False
    path: Path | None = field(default=None, compare=False)

    def render(self, max_chars: int = 2500) -> str:
        """The text that goes in the prompt. Clipped, because a brief that runs
        long crowds out the lessons block rather than adding to it."""
        body = "\n".join(line.rstrip() for line in self.body.strip().splitlines())
        if len(body) > max_chars:
            body = body[: max_chars - 1].rstrip() + "…"
        return body


def parse_brief(text: str, path: Path | None = None) -> Brief:
    m = _FRONT_MATTER.match(text)
    if not m:
        raise BriefError("no YAML front matter (file must start with a '---' line)")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise BriefError(f"unreadable front matter: {e}") from e
    if not isinstance(meta, dict):
        raise BriefError("front matter must be a mapping")

    missing = [k for k in _REQUIRED if not str(meta.get(k) or "").strip()]
    if missing:
        raise BriefError(f"missing required field(s): {', '.join(missing)}")

    brief_id = str(meta["id"]).strip()
    if not _ID_RE.match(brief_id):
        raise BriefError(
            f"id {brief_id!r} must be lowercase kebab-case, 3-64 chars — it is "
            "stored as provenance on every experiment it produces"
        )

    body = m.group(2).strip()
    if not body:
        raise BriefError("empty body — front matter alone tells the model nothing")

    sources = meta.get("sources") or []
    if isinstance(sources, str):
        sources = [sources]

    return Brief(
        id=brief_id,
        title=str(meta["title"]).strip(),
        family=str(meta["family"]).strip(),
        body=body,
        sources=tuple(str(s).strip() for s in sources if str(s).strip()),
        retired=bool(meta.get("retired", False)),
        path=path,
    )


class BriefLibrary:
    """The brief directory, plus the policy for which one to hand out next.

    Selection is least-attempted-first and deterministic, so a session works
    through the library in a predictable rotation instead of re-rolling the same
    idea. A brief that has been attempted `max_attempts` times is retired from
    the rotation: it has had its shot, and continuing to re-code an idea that
    keeps dying is how a search burns a night on one hypothesis.
    """

    def __init__(self, directory: str | Path, *, max_attempts: int = 3):
        self.directory = Path(directory)
        self.max_attempts = max_attempts
        self.briefs: list[Brief] = []
        self.all_briefs: list[Brief] = []
        # Malformed files are collected rather than raised: one bad brief should
        # not end an unattended session. They are surfaced loudly by the caller,
        # because a research pipeline that has silently stopped supplying ideas
        # looks exactly like one that is working.
        self.problems: list[str] = []
        self.load()

    def load(self) -> None:
        # `briefs` is the rotation; `all_briefs` also keeps the retired ones, so
        # the dashboard can show a brief that has been taken out of service
        # rather than having it silently vanish from the library.
        self.briefs, self.all_briefs, self.problems = [], [], []
        if not self.directory.is_dir():
            return
        seen: dict[str, Path] = {}
        for path in sorted(self.directory.glob("*.md")):
            try:
                brief = parse_brief(path.read_text(encoding="utf-8"), path=path)
            except (BriefError, OSError) as e:
                self.problems.append(f"{path.name}: {e}")
                continue
            if brief.id in seen:
                self.problems.append(
                    f"{path.name}: duplicate id {brief.id!r} (also in {seen[brief.id].name})"
                )
                continue
            seen[brief.id] = path
            self.all_briefs.append(brief)
            if not brief.retired:
                self.briefs.append(brief)

    def by_id(self, brief_id: str) -> Brief | None:
        return next((b for b in self.briefs if b.id == brief_id), None)

    def select(self, attempts: dict[str, int] | None = None) -> Brief | None:
        """The next brief to code, or None when the library is exhausted.

        None is a normal outcome, not an error: the loop falls back to asking the
        model to invent something, which is what it did before briefs existed.
        """
        attempts = attempts or {}
        live = [b for b in self.briefs if attempts.get(b.id, 0) < self.max_attempts]
        if not live:
            return None
        return min(live, key=lambda b: (attempts.get(b.id, 0), b.id))
