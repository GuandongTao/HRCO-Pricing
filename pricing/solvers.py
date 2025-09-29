from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
from scipy import optimize

@dataclass(frozen=True)
class Rootsettings:
    tol: float = 1e-8
    maxiter: int = 50
    a: float = 1e-6
    b: float = 5.0

class RootSolver:
    def __init__(self, s: Rootsettings = Rootsettings()):
        self.s = s

    def newton(self, f: Callable[[float], float], x0: float,
               fprime: Optional[Callable[[float], float]]) -> float:
        """Newton/Secant via Scipy"""
        return optimize.newton(f=f, x0=x0, fprime=fprime,
                               tol=self.s.tol, maxiter=self.s.maxiter)

    def brentq(self, f: Callable[[float], float],
               a: Optional[float] = None, b: Optional[float] = None) -> float:
        """Safe bracketed root"""
        a = self.s.a if a is None else a
        b = self.s.b if b is None else b
        return optimize.brentq(f, a, b, xtol=self.s.tol, maxiter=self.s.maxiter)