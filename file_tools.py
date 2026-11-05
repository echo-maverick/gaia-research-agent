
"""File-processing tools for the GAIA research agent."""

import re
from decimal import Decimal
from pathlib import Path

import openpyxl


def transcribe_audio(file_path, model_name="small"):
    """Transcribe an audio file using Whisper."""
    import torch
    import whisper

    if not Path(file_path).is_file():
        raise FileNotFoundError(file_path)

    model = whisper.load_model(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    result = model.transcribe(
        str(file_path),
        fp16=torch.cuda.is_available()
    )

    return result["text"].strip()


def extract_page_numbers(transcript):
    """Extract page references from spoken instructions."""
    matches = re.findall(
        r"\bpages?\s+([\d,\s]+(?:and\s+\d+)?)",
        transcript,
        flags=re.IGNORECASE
    )

    numbers = set()

    for match in matches:
        numbers.update(
            int(value)
            for value in re.findall(r"\d+", match)
        )

    return ", ".join(
        str(number) for number in sorted(numbers)
    )


def extract_ingredients(transcript):
    """Extract ingredients from a simple spoken recipe."""
    ingredients = []

    combine = re.search(
        r"\bcombine\s+(.+?)(?:\.\s|$)",
        transcript,
        flags=re.IGNORECASE
    )

    if combine:
        text = re.sub(
            r"\s+and\s+", ", ",
            combine.group(1)
        )
        ingredients.extend(text.split(","))

    additions = re.findall(
        r"\bstir in\s+(.*?)(?:\.|$)",
        transcript,
        flags=re.IGNORECASE
    )

    for item in additions:
        item = re.sub(
            r"^(a dash of|a pinch of|some)\s+",
            "",
            item,
            flags=re.IGNORECASE
        )
        ingredients.append(item)

    return ", ".join(
        sorted(
            (item.strip() for item in ingredients if item.strip()),
            key=str.lower
        )
    )


def calculate_food_sales(file_path):
    """Sum approved food columns in a sales workbook."""
    workbook = openpyxl.load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )

    excluded = {
        "location", "store",
        "soda", "drinks", "beverages",
        "coffee", "tea"
    }

    total = Decimal("0")

    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows, None)

            if headers is None:
                continue

            food_columns = [
                index
                for index, header in enumerate(headers)
                if header
                and str(header).strip().lower() not in excluded
            ]

            for row in rows:
                for index in food_columns:
                    value = row[index]

                    if isinstance(value, (int, float)):
                        total += Decimal(str(value))
    finally:
        workbook.close()

    return f"${total:,.2f}"
    
def olympic_smallest_team(url):
    """Find the smallest participating team, breaking ties by country name."""
    import requests
    import pandas as pd
    from io import StringIO

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    for table in tables:
        if {"Country", "Total"}.issubset(table.columns):
            data = table.copy()
            data["Total"] = pd.to_numeric(
                data["Total"], errors="coerce"
            )
            data = data.dropna(subset=["Country", "Total"])
            ranked = data.sort_values(["Total", "Country"])
            return str(ranked.iloc[0, 0])

    raise ValueError("Participation table not found")


def find_noncommutative_elements(elements, operation):
    """Return elements involved in noncommutative pairs."""
    involved = set()

    for a in elements:
        for b in elements:
            if operation[a][b] != operation[b][a]:
                involved.update((a, b))

    return ", ".join(sorted(involved))

def solve_commutativity_table(question):
    """Find elements in noncommutative pairs from a Markdown table."""
    import re

    lines = [
        line.strip()
        for line in question.splitlines()
        if line.strip().startswith("|")
    ]

    rows = []
    for line in lines:
        cells = [x.strip() for x in line.strip("|").split("|")]

        if all(re.fullmatch(r":?-+:?", x) for x in cells):
            continue

        rows.append(cells)

    if len(rows) < 2:
        raise ValueError("No valid operation table found")

    elements = rows[0][1:]

    if len(rows) != len(elements) + 1:
        raise ValueError("Incomplete operation table")

    operation = {
        row[0]: dict(zip(elements, row[1:]))
        for row in rows[1:]
    }

    involved = set()

    for a in elements:
        for b in elements:
            if operation[a][b] != operation[b][a]:
                involved.update([a, b])

    return ", ".join(sorted(involved))


