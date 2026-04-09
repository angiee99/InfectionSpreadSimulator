import numpy as np


class InfectionSimulator:
    @staticmethod
    def create_initial_state(num_nodes: int, infected_nodes: list[int]) -> np.ndarray:
        state = np.zeros(num_nodes, dtype=int)
        for node in infected_nodes:
            if node < 0 or node >= num_nodes:
                raise ValueError(f"Invalid infected node index: {node}")
            state[node] = 1
        return state

    @staticmethod
    def deterministic_step(A: np.ndarray, state: np.ndarray) -> np.ndarray:
        influence = A @ state
        next_state = (influence > 0).astype(int)
        return next_state

    @staticmethod
    def probabilistic_step(
        A: np.ndarray,
        state: np.ndarray,
        beta: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        influence = A @ state
        probs = 1.0 - (1.0 - beta) ** influence
        random_values = rng.random(len(state))
        next_state = (random_values < probs).astype(int)
        return next_state

    @classmethod
    def run_deterministic_history(
        cls,
        A: np.ndarray,
        initial_state: np.ndarray,
        steps: int,
    ) -> list[np.ndarray]:
        history = [initial_state.copy()]
        current = initial_state.copy()

        for _ in range(steps):
            current = cls.deterministic_step(A, current)
            history.append(current.copy())

        return history

    @classmethod
    def run_probabilistic_history(
        cls,
        A: np.ndarray,
        initial_state: np.ndarray,
        steps: int,
        beta: float,
        random_seed: int,
    ) -> list[np.ndarray]:
        history = [initial_state.copy()]
        current = initial_state.copy()
        rng = np.random.default_rng(random_seed)

        for _ in range(steps):
            current = cls.probabilistic_step(A, current, beta=beta, rng=rng)
            history.append(current.copy())

        return history