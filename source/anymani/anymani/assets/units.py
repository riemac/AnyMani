"Provides authoring-side conversion helpers. Builders and schemas store plain floats in SI units: meters, radians, kilograms, and kilograms per cubic meter."

from __future__ import annotations

import math


def m(value: float) -> float:
    "Returns meters as the length unit."

    return float(value)


def cm(value: float) -> float:
    "Converts centimeters to meters."

    return float(value) / 100.0


def mm(value: float) -> float:
    "Converts millimeters to meters."

    return float(value) / 1000.0


def rad(value: float) -> float:
    "Returns radians as the angle unit."

    return float(value)


def deg(value: float) -> float:
    "Converts degrees to radians."

    return math.radians(float(value))


def kg(value: float) -> float:
    "Returns kilograms as the mass unit."

    return float(value)


def g(value: float) -> float:
    "Converts grams to kilograms."

    return float(value) / 1000.0


def kg_m3(value: float) -> float:
    "Returns kilograms per cubic meter as the density unit."

    return float(value)


def g_cm3(value: float) -> float:
    "Converts grams per cubic centimeter to kilograms per cubic meter."

    return float(value) * 1000.0


__all__ = [
    "m",
    "cm",
    "mm",
    "rad",
    "deg",
    "kg",
    "g",
    "kg_m3",
    "g_cm3",
]
