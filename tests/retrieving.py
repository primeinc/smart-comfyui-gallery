"""One shape for a faked `retrieval.query` answer.

Six test modules independently hand-rolled the dict retrieval returns,
so widening that contract by two keys broke six files that had no
opinion about the new keys. The shape belongs in one place: a double
that has to be edited in six places to stay a double is six chances to
leave one of them describing an answer the application no longer gives.

`answered` takes the file ids a fake ranking produced and returns what
db/resultset.py will read. Its default is "every one of these answers",
because a test about scope, order, locking or provenance is not a test
about where the ranking stops -- that one is
tests/test_a_search_is_a_set.py, and it uses real scores.
"""

from __future__ import annotations

from collections.abc import Iterable


def answered(
    ids: Iterable[int],
    *,
    participants: list[str] | None = None,
    contributors: list[str] | None = None,
    missing: dict[str, str] | None = None,
    unmatched: dict[str, str] | None = None,
    answers: bool = True,
) -> dict:
    """A retrieval answer over `ids`, in that order.

    `answers` is retrieval's verdict on whether a file stands above the
    middle of what a space said (db/retrieval.py `head`). True here
    means the double asserts nothing about the cut, leaving whatever
    the test IS about as the only thing narrowing the answer.
    """
    held = list(ids)
    return {
        "results": [
            {
                "file_id": file_id,
                "score": 1.0,
                "relevance": 1.0 if answers else 0.0,
                "answers": answers,
                "sources": {},
            }
            for file_id in held
        ],
        "participants": participants if participants is not None else ["fake"],
        "contributors": contributors if contributors is not None else ["fake"],
        "missing": missing or {},
        "unmatched": unmatched or {},
        "answering": len(held) if answers else 0,
    }
