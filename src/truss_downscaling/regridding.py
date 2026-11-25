from __future__ import annotations

from typing import Any

from .config import Config

#: Regridding methods exposed by xESMF that make sense for this contract.
#:
#: ``bilinear``           smooth interpolation; preserves the mean, damps variance
#:                        and both tails. The historical behaviour.
#: ``conservative``       first-order conservative; every target cell takes the
#:                        source cell's value, so the coarse field's own intensity
#:                        survives (blocky, mass preserving).
#: ``conservative_normed`` conservative with per-target-cell normalisation.
#: ``nearest_s2d``        nearest source to destination; like conservative for
#:                        upscaling but copies the nearest cell rather than
#:                        overlapping weights.
#: ``patch``              patch (higher order) interpolation; smoother than bilinear,
#:                        still not mass preserving.
SUPPORTED_METHODS: tuple[str, ...] = (
    "bilinear",
    "conservative",
    "conservative_normed",
    "nearest_s2d",
    "patch",
)

DEFAULT_METHOD = "bilinear"

#: Regridding methods where each target cell inherits a source cell value instead
#: of an average of neighbours. Used to describe the choice in provenance.
NON_SMOOTHING_METHODS = frozenset({"conservative", "conservative_normed", "nearest_s2d"})


def _check_method(method: Any, where: str) -> str:
    if not isinstance(method, str):
        raise ValueError(f"{where} must be a string, got {type(method).__name__}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"unsupported regridding method {method!r} in {where}; "
            f"choose one of {', '.join(SUPPORTED_METHODS)}"
        )
    return method


def resolve_regrid_methods(config: Config) -> dict[str, str]:
    """Map each input channel's source variable to its xESMF regridding method.

    Reads ``preprocessing.regridding`` (the default, ``bilinear`` when absent) and
    ``preprocessing.regridding_by_variable`` (per-variable overrides, keyed by the
    source variable name without the ``_gcm`` suffix).
    """
    block = config.raw.get("preprocessing") or {}
    if not isinstance(block, dict):
        raise ValueError("preprocessing must be a mapping")
    default = _check_method(block.get("regridding", DEFAULT_METHOD), "preprocessing.regridding")
    overrides = block.get("regridding_by_variable") or {}
    if not isinstance(overrides, dict):
        raise ValueError("preprocessing.regridding_by_variable must be a mapping")

    channels = list(config.data.get("input_channels") or [])
    if not channels:
        raise ValueError("data.input_channels must list at least one channel")
    variables = [str(channel).removesuffix("_gcm") for channel in channels]

    unknown = sorted(set(overrides) - set(variables))
    if unknown:
        raise ValueError(
            "preprocessing.regridding_by_variable names variables that are not input "
            f"channels: {', '.join(unknown)}"
        )

    resolved: dict[str, str] = {}
    for variable in variables:
        method = overrides.get(variable, default)
        resolved[variable] = _check_method(
            method, f"preprocessing.regridding_by_variable[{variable!r}]"
        )
    return resolved


def describe_regridding(methods: dict[str, str]) -> dict[str, Any]:
    """Provenance record for a resolved method map."""
    distinct = sorted(set(methods.values()))
    uniform = len(distinct) == 1
    return {
        "label": f"xesmf_{distinct[0]}" if uniform else "xesmf_mixed",
        "uniform": uniform,
        "distinct": distinct,
        "by_variable": dict(sorted(methods.items())),
        "smoothing": {
            variable: method not in NON_SMOOTHING_METHODS
            for variable, method in sorted(methods.items())
        },
    }


def regrid_methods_for(norm: dict[str, Any], config: Config) -> dict[str, str]:
    """Methods to use at inference.

    The preprocessing record wins, because a checkpoint must be served with the
    interpolation it was trained on. An explicit config that disagrees is an error
    rather than a silent train/serve skew; a config that says nothing is fine.
    """
    recorded = norm.get("regrid_methods")
    block = config.raw.get("preprocessing") or {}
    explicit = "regridding" in block or "regridding_by_variable" in block
    if recorded:
        recorded = {str(k): str(v) for k, v in recorded.items()}
        for variable, method in recorded.items():
            _check_method(method, f"normalization.regrid_methods[{variable!r}]")
        if explicit:
            configured = resolve_regrid_methods(config)
            if configured != recorded:
                raise ValueError(
                    "inference config regridding disagrees with the preprocessing "
                    f"artifact the checkpoint was trained with: config={configured} "
                    f"artifact={recorded}"
                )
        return recorded
    return resolve_regrid_methods(config)
