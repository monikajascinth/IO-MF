from dataclasses import dataclass
from math import exp, fsum, isfinite, log
from typing import Protocol, Sequence


class Inference(Protocol):
    def encode(self, modality: str) -> object: ...
    def probe(self, modality: str) -> object: ...
    def logits(self, embeddings: dict[str, object]) -> Sequence[float]: ...


@dataclass(frozen=True)
class Timing:
    primary: int
    probe: dict[str, int]
    commit: dict[str, int]


@dataclass(frozen=True)
class Event:
    action: str
    modality: str
    residual: int | None
    utility: float | None = None
    entropy: float | None = None


@dataclass(frozen=True)
class Result:
    active: tuple[str, ...]
    probabilities: tuple[float, ...] | None
    residual: int | None
    reason: str
    trace: tuple[Event, ...]

    @property
    def prediction(self):
        if self.probabilities is None:
            return None
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)


def distribution(logits):
    values = tuple(float(value) for value in logits)
    if not values or not all(map(isfinite, values)):
        raise ValueError("Expected nonempty finite logits")
    peak = max(values)
    weights = [exp(value - peak) for value in values]
    total = fsum(weights)
    return tuple(weight / total for weight in weights)


def entropy(probabilities):
    return fsum(-p * log(p) for p in probabilities if p > 0)


def infer(engine, primary, optional, costs, weight, *, deadline=None, timing=None):
    """Run Algorithm 1, or Algorithm 2 with integer time bounds in a common unit."""
    primary, optional = tuple(primary), tuple(optional)
    modalities = primary + optional
    if not primary or len(set(modalities)) != len(modalities):
        raise ValueError("Provide a nonempty primary set and unique modality IDs")
    if not isfinite(weight) or weight < 0 or any(
        not isfinite(costs[m]) or costs[m] < 0 for m in optional
    ):
        raise ValueError("Utility weight and encoder costs must be finite and nonnegative")
    if (deadline is None) != (timing is None):
        raise ValueError("Deadline and timing bounds must be supplied together")

    residual = None
    if timing is not None:
        bounds = [timing.primary, *(timing.probe[m] for m in optional),
                  *(timing.commit[m] for m in optional)]
        if type(deadline) is not int or deadline <= 0 or any(
            type(bound) is not int or bound < 0 for bound in bounds
        ):
            raise ValueError("Use a positive integer deadline and nonnegative integer bounds")
        residual = deadline - timing.primary
        if residual < 0:
            return Result((), None, deadline, "infeasible", ())

    cache = {m: engine.encode(m) for m in primary}
    probabilities = distribution(engine.logits(cache))
    trace = [Event("primary", m, residual) for m in primary]
    remaining = list(optional)
    reason = "exhausted"
    while remaining:
        leader = None
        best_score = 0.0
        probed = set()
        current_entropy = entropy(probabilities)
        for modality in remaining:
            if timing is not None:
                probe_time = timing.probe[modality]
                reservation = timing.commit[leader] if leader is not None else 0
                if probe_time + max(timing.commit[modality], reservation) > residual:
                    trace.append(Event("skip", modality, residual))
                    continue
                residual -= probe_time
            surrogate = engine.probe(modality)
            candidate_entropy = entropy(distribution(engine.logits({**cache, modality: surrogate})))
            score = current_entropy - candidate_entropy - weight * costs[modality]
            trace.append(Event("probe", modality, residual, score, candidate_entropy))
            probed.add(modality)
            if score > best_score:
                leader, best_score = modality, score

        if leader is None:
            reason = "no_positive_utility" if probed else "no_admissible_probe"
            break

        cache[leader] = engine.encode(leader)
        probabilities = distribution(engine.logits(cache))
        if timing is not None:
            residual -= timing.commit[leader]
        trace.append(Event("commit", leader, residual, best_score, entropy(probabilities)))
        if timing is None:
            remaining.remove(leader)
        else:
            remaining = [modality for modality in remaining if modality not in probed]

    return Result(tuple(cache), probabilities, residual, reason, tuple(trace))
