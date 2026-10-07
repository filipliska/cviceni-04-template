# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Abstraktní třída vzdálenosti a její konkrétní implementace.
"""

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """Abstraktní základ pro metriky vzdálenosti."""

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí ``True``, pokud vzdálenost splňuje axiomy metriky."""

    @abstractmethod
    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá vzdálenost mezi dvěma body."""
        assert point_a.ndim == 1, "point_a musí být 1D vektor"
        assert point_b.ndim == 1, "point_b musí být 1D vektor"
        assert len(point_a) == len(point_b), (
            "point_a a point_b musí mít stejnou délku"
        )

        raise NotImplementedError

    def create_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Vytvoří čtvercovou matici vzdáleností mezi všemi dvojicemi bodů."""

        assert data.ndim == 2, "data musí být 2D matice"
        assert data.shape[0] >= 2, "data musí obsahovat alespoň 2 body"

        n: int = data.shape[0]
        matrix: np.ndarray = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                dist: float = self.calculate(data[i], data[j])
                matrix[i, j] = dist
                matrix[j, i] = dist

        return matrix


class EuclideanDistance(Distance):
    """Euklidovská vzdálenost — délka přímé spojnice dvou bodů."""

    @property
    def is_metric(self) -> bool:
        """Euklidovská vzdálenost je pravá metrika."""
        return True

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá euklidovskou vzdálenost mezi dvěma body."""

        assert point_a.ndim == 1, "point_a musí být 1D vektor"
        assert point_b.ndim == 1, "point_b musí být 1D vektor"
        assert len(point_a) == len(point_b), (
            "point_a a point_b musí mít stejnou délku"
        )

        return float(np.sqrt(np.sum((point_a - point_b) ** 2)))


class ManhattanDistance(Distance):
    """Manhattanská vzdálenost — součet absolutních rozdílů souřadnic."""

    @property
    def is_metric(self) -> bool:
        """Manhattanská vzdálenost je pravá metrika."""
        return True

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá manhattanskou vzdálenost mezi dvěma body."""

        assert point_a.ndim == 1, "point_a musí být 1D vektor"
        assert point_b.ndim == 1, "point_b musí být 1D vektor"
        assert len(point_a) == len(point_b), (
            "point_a a point_b musí mít stejnou délku"
        )

        return float(np.sum(np.abs(point_a - point_b)))


class CosineCoeficient(Distance):
    """Kosinová podobnost (jako vzdálenost: 1 - kosinová_podobnost)."""

    @property
    def is_metric(self) -> bool:
        """Kosinová vzdálenost není pravá metrika."""
        return False

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá kosinovou vzdálenost mezi dvěma body."""

        assert point_a.ndim == 1, "point_a musí být 1D vektor"
        assert point_b.ndim == 1, "point_b musí být 1D vektor"
        assert len(point_a) == len(point_b), (
            "point_a a point_b musí mít stejnou délku"
        )

        norm_a = np.linalg.norm(point_a)
        norm_b = np.linalg.norm(point_b)

        # Oba vektory jsou nulové → považujeme je za identické.
        if norm_a == 0 and norm_b == 0:
            return 0.0

        # Právě jeden vektor je nulový → kosinová vzdálenost = 1.
        if norm_a == 0 or norm_b == 0:
            return 1.0

        cosine_similarity = np.dot(point_a, point_b) / (norm_a * norm_b)

        return float(1.0 - cosine_similarity)
