import random

from toysched.placement import place


def test_keeps_held_slots():
    new, changes = place(["a", "a", "b", None], {"a": 1, "b": 2, "c": 1})
    assert new[0] == "a" and new[2] == "b"
    assert sorted(x for x in new if x) == ["a", "b", "b", "c"]
    assert changes == 2  # b grows by 1, c by 1


def test_unchanged_counts_change_nothing():
    prev = ["a", "b", "a", None]
    new, changes = place(prev, {"a": 2, "b": 1})
    assert new == prev and changes == 0


def test_changes_meet_lower_bound_randomized():
    rng = random.Random(0)
    tasks = list("abcdef")
    for _ in range(1000):
        S = rng.randint(1, 10)
        prev = [rng.choice(tasks + [None]) for _ in range(S)]
        budget = S
        counts = {}
        for t in rng.sample(tasks, rng.randint(0, len(tasks))):
            n = rng.randint(0, budget)
            if n:
                counts[t] = n
                budget -= n
        new, changes = place(prev, counts)
        for t in tasks:
            assert new.count(t) == counts.get(t, 0)
        lower = sum(max(0, n - prev.count(t)) for t, n in counts.items())
        assert changes == lower
        moved = sum(1 for p, n in zip(prev, new) if n is not None and n != p)
        assert moved == lower
