"""Per-stage calculator selection for the solvation free-energy workflow.

The workflow is split into four independently configurable stages:

- ``gas``: gas-phase single points on embedded (screening) conformers, plus the
  post-solvation gas-phase re-optimization and single point.
- ``freq``: Hessian / RRHO free-energy correction.
- ``solvent_opt``: geometry optimization in implicit solvent.
- ``solvation``: implicit-solvation free energy (dG_solv) single point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, get_args

XtbVersion = Literal["legacy", "default"]
GasCalculator = Literal["gxtb", "gfn2", "aimnet2", "orca"]
FreqCalculator = Literal["gfn2", "gxtb", "orca"]
SolventOptCalculator = Literal["gfn2", "aimnet2", "orca"]
SolvationCalculator = Literal["cpcmx", "orca"]
OrcaSolventOptModel = Literal["cpcm", "smd"]
OrcaSolvationModel = Literal["cpcm", "smd", "cosmors"]

GAS_CALCULATORS: tuple[str, ...] = get_args(GasCalculator)
FREQ_CALCULATORS: tuple[str, ...] = get_args(FreqCalculator)
SOLVENT_OPT_CALCULATORS: tuple[str, ...] = get_args(SolventOptCalculator)
SOLVATION_CALCULATORS: tuple[str, ...] = get_args(SolvationCalculator)
ORCA_SOLVENT_OPT_MODELS: tuple[str, ...] = get_args(OrcaSolventOptModel)
ORCA_SOLVATION_MODELS: tuple[str, ...] = get_args(OrcaSolvationModel)

_ORCA_MODEL_LABELS = {"cpcm": "CPCM", "smd": "SMD", "cosmors": "openCOSMO-RS"}


@dataclass(frozen=True)
class OrcaSettings:
    """Shared ORCA runtime settings applied to every ORCA job."""

    executable: str = "orca"
    nprocs: int = 1
    maxcore_mb: Optional[int] = None
    extra_input: Optional[str] = None
    solvent: Optional[str] = None


@dataclass(frozen=True)
class CalculatorSettings:
    """Calculator backend selected for each workflow stage."""

    gas: GasCalculator = "gxtb"
    freq: FreqCalculator = "gfn2"
    solvent_opt: SolventOptCalculator = "gfn2"
    solvation: SolvationCalculator = "cpcmx"
    xtb_executable: str = "xtb"
    xtb_version: XtbVersion = "default"
    orca: OrcaSettings = field(default_factory=OrcaSettings)
    orca_keywords: Optional[str] = None
    gas_orca_keywords: Optional[str] = None
    freq_orca_keywords: Optional[str] = None
    solvent_opt_orca_keywords: Optional[str] = None
    solvent_opt_orca_model: OrcaSolventOptModel = "cpcm"
    solvation_orca_keywords: Optional[str] = None
    solvation_orca_model: OrcaSolvationModel = "cpcm"

    def resolved_orca_keywords(self, stage: str) -> Optional[str]:
        """Stage-specific ORCA keywords, falling back to the shared ``orca_keywords``."""
        per_stage = {
            "gas": self.gas_orca_keywords,
            "freq": self.freq_orca_keywords,
            "solvent_opt": self.solvent_opt_orca_keywords,
            "solvation": self.solvation_orca_keywords,
        }[stage]
        value = per_stage if per_stage is not None else self.orca_keywords
        if value is None or not value.strip():
            return None
        return value.strip()

    def solvation_orca_job_keywords(self) -> Optional[str]:
        """Keywords for the ORCA solvation job.

        openCOSMO-RS fixes its own level of theory, so only explicit
        ``solvation_orca_keywords`` are used there (no shared fallback).
        """
        if self.solvation_orca_model == "cosmors":
            value = self.solvation_orca_keywords
            return value.strip() if value and value.strip() else None
        return self.resolved_orca_keywords("solvation")

    def require_orca_keywords(self, stage: str) -> str:
        keywords = self.resolved_orca_keywords(stage)
        if keywords is None:
            raise ValueError(f"No ORCA keywords configured for the {stage} stage.")
        return keywords

    @property
    def uses_xtb(self) -> bool:
        return (
            self.gas in ("gxtb", "gfn2")
            or self.freq in ("gfn2", "gxtb")
            or self.solvent_opt == "gfn2"
            or self.solvation == "cpcmx"
        )

    @property
    def uses_orca(self) -> bool:
        return "orca" in (self.gas, self.freq, self.solvent_opt, self.solvation)

    def validate(self) -> None:
        """Raise ValueError when the stage configuration is incomplete or inconsistent."""
        checks = (
            ("gas", self.gas, GAS_CALCULATORS),
            ("freq", self.freq, FREQ_CALCULATORS),
            ("solvent_opt", self.solvent_opt, SOLVENT_OPT_CALCULATORS),
            ("solvation", self.solvation, SOLVATION_CALCULATORS),
        )
        for stage, value, allowed in checks:
            if value not in allowed:
                raise ValueError(f"Unknown {stage} calculator {value!r}; choose from {', '.join(allowed)}.")
        if self.solvent_opt_orca_model not in ORCA_SOLVENT_OPT_MODELS:
            raise ValueError(f"Unknown ORCA solvent-optimization model {self.solvent_opt_orca_model!r}.")
        if self.solvation_orca_model not in ORCA_SOLVATION_MODELS:
            raise ValueError(f"Unknown ORCA solvation model {self.solvation_orca_model!r}.")
        if self.orca.nprocs < 1:
            raise ValueError("ORCA nprocs must be >= 1.")
        if self.orca.maxcore_mb is not None and self.orca.maxcore_mb < 1:
            raise ValueError("ORCA maxcore must be >= 1 MB.")

        for stage, value in (("gas", self.gas), ("freq", self.freq), ("solvent_opt", self.solvent_opt)):
            if value == "orca" and self.resolved_orca_keywords(stage) is None:
                raise ValueError(
                    f"The {stage} stage uses ORCA but no ORCA keywords were given; "
                    f"set --orca-keywords or --{_STAGE_CLI_PREFIX[stage]}-orca-keywords."
                )
        if (
            self.solvation == "orca"
            and self.solvation_orca_model != "cosmors"
            and self.resolved_orca_keywords("solvation") is None
        ):
            raise ValueError(
                "The solvation stage uses ORCA CPCM/SMD but no ORCA keywords were given; "
                "set --orca-keywords or --solv-sp-orca-keywords."
            )

    def label(self, stage: str) -> str:
        """Short human-readable level-of-theory label for progress/log messages."""
        if stage == "gas":
            return {
                "gxtb": "g-xTB",
                "gfn2": "GFN2-xTB",
                "aimnet2": "AIMNet2",
            }.get(self.gas) or f"ORCA {self.resolved_orca_keywords('gas')}"
        if stage == "freq":
            return {
                "gfn2": "GFN2-xTB",
                "gxtb": "g-xTB",
            }.get(self.freq) or f"ORCA {self.resolved_orca_keywords('freq')}"
        if stage == "solvent_opt":
            if self.solvent_opt == "gfn2":
                return "GFN2-xTB/ALPB"
            if self.solvent_opt == "aimnet2":
                return "AIMNet2 (gas phase)"
            model = _ORCA_MODEL_LABELS[self.solvent_opt_orca_model]
            return f"ORCA {self.resolved_orca_keywords('solvent_opt')}/{model}"
        if stage == "solvation":
            if self.solvation == "cpcmx":
                return "CPCM-X"
            model = _ORCA_MODEL_LABELS[self.solvation_orca_model]
            keywords = self.solvation_orca_job_keywords()
            return f"ORCA {model}" if keywords is None else f"ORCA {keywords}/{model}"
        raise ValueError(f"Unknown workflow stage: {stage}")

    def summary(self) -> str:
        return (
            f"gas={self.label('gas')}; freq={self.label('freq')}; "
            f"solvent_opt={self.label('solvent_opt')}; solvation={self.label('solvation')}"
        )


_STAGE_CLI_PREFIX = {
    "gas": "gas",
    "freq": "freq",
    "solvent_opt": "solv-opt",
    "solvation": "solv-sp",
}
