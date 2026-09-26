"""The renderer-facing recipe every environment family expands into."""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Sun:
    """A distant light: direction toward the sun, irradiance and colour."""

    elevation: float  # degrees above the horizon
    azimuth: float  # degrees counter-clockwise from +X, toward the sun
    strength: float  # W/m^2, Blender sun strength
    angle: float  # apparent diameter in degrees; larger is softer
    color: RGB  # scene-linear, max channel 1


@dataclass(frozen=True)
class Sky:
    """A physical clear sky (Blender's scattering model) without its sun disc."""

    elevation: float
    azimuth: float
    haze: float  # aerosol density, 0..10
    strength: float  # background multiplier applied to the physical sky


@dataclass(frozen=True)
class EnvironmentRecipe:
    """Everything that lights a scene from outside it.

    The world background is ``sky`` (if any) plus a uniform ``fill`` colour at
    ``fill_strength`` above the horizon and ``ground`` below it; ``sun`` adds a
    matching distant light.  Families keep these
    consistent with each other, which a model assembling lights by hand rarely does.
    """

    fill: RGB = (0.0, 0.0, 0.0)
    fill_strength: float = 0.0
    sky: Sky | None = None
    sun: Sun | None = None
    #: Radiance of the ground below the horizon (scene-linear); None means the
    #: background is the same above and below the horizon.
    ground: RGB | None = None

    def __post_init__(self) -> None:
        if self.fill_strength < 0:
            raise ValueError("fill_strength must be non-negative")
        if self.sun is not None and self.sun.strength < 0:
            raise ValueError("sun strength must be non-negative")
        if self.sky is not None and self.sky.strength < 0:
            raise ValueError("sky strength must be non-negative")

    @property
    def lit(self) -> bool:
        return (
            (self.fill_strength > 0 and max(self.fill) > 0)
            or (self.sky is not None and self.sky.strength > 0)
            or (self.sun is not None and self.sun.strength > 0)
        )
