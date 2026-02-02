#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Script to generate data source summary tables for documentation."""

import inspect
from earth2studio.data.base import (
    DataSource,
    ForecastSource,
    DataFrameSource,
    ForecastFrameSource,
)
import earth2studio.data as data_module


def get_first_docstring_line(obj) -> str:
    """Get the first line of a docstring."""
    doc = inspect.getdoc(obj)
    if doc:
        return doc.split("\n")[0]
    return ""


def classify_data_sources():
    """Classify all exported data sources by their interface type."""
    datasources = []
    forecastsources = []
    dataframesources = []
    forecastframesources = []
    utilities = []

    # Get all public members from the data module
    for name in dir(data_module):
        if name.startswith("_"):
            continue

        obj = getattr(data_module, name)

        # Skip non-classes and base protocols
        if not inspect.isclass(obj):
            if callable(obj):
                utilities.append((name, get_first_docstring_line(obj)))
            continue

        if name in ("DataSource", "ForecastSource", "DataFrameSource", "ForecastFrameSource"):
            continue

        # Check which protocol it implements
        try:
            if isinstance(obj, type):
                # Check by method signatures
                has_lead_time = False
                returns_dataframe = False

                if hasattr(obj, "__call__"):
                    sig = inspect.signature(obj.__call__)
                    params = list(sig.parameters.keys())
                    has_lead_time = "lead_time" in params

                    # Check return annotation
                    if sig.return_annotation != inspect.Signature.empty:
                        ret = str(sig.return_annotation)
                        returns_dataframe = "DataFrame" in ret

                desc = get_first_docstring_line(obj)

                if returns_dataframe and has_lead_time:
                    forecastframesources.append((name, desc))
                elif returns_dataframe:
                    dataframesources.append((name, desc))
                elif has_lead_time:
                    forecastsources.append((name, desc))
                else:
                    datasources.append((name, desc))
        except Exception:
            datasources.append((name, get_first_docstring_line(obj)))

    return {
        "DataSource": datasources,
        "ForecastSource": forecastsources,
        "DataFrameSource": dataframesources,
        "ForecastFrameSource": forecastframesources,
        "Utilities": utilities,
    }


def generate_table(items: list[tuple[str, str]], header: str = "Source") -> str:
    """Generate a markdown table from items."""
    if not items:
        return "*No sources available*\n"

    lines = [
        f"| {header} | Description |",
        "|--------|-------------|",
    ]
    for name, desc in sorted(items):
        lines.append(f"| `{name}` | {desc} |")

    return "\n".join(lines) + "\n"


def main():
    classified = classify_data_sources()

    print("## Data Sources (Analysis/Reanalysis)\n")
    print(generate_table(classified["DataSource"]))

    print("\n## Forecast Sources\n")
    print(generate_table(classified["ForecastSource"]))

    print("\n## DataFrame Sources\n")
    print(generate_table(classified["DataFrameSource"]))

    print("\n## Utility Functions\n")
    print(generate_table(classified["Utilities"], header="Function"))


if __name__ == "__main__":
    main()
