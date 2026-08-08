import numpy as np

DEFAULT_SEED = 42

# Named RNG streams within a run. Keeping these separate means changing one
# component (e.g., weather) does not perturb another (e.g., order arrivals),
# which makes cross-scenario comparisons clean.
STREAMS = ["arrivals", "weather", "traffic", "cancellations", "prep"]


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Create a fresh, seeded numpy RNG for a simulation run."""
    return np.random.default_rng(seed)


def spawn_streams(seed: int = DEFAULT_SEED) -> dict[str, np.random.Generator]:
    """Create one independent, seeded RNG per simulation component."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}
