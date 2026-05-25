# Bilig WorkPaper Formula Readback

Use a Bilig WorkPaper endpoint from Open WebUI to write an input cell, recalculate formulas, and return verified JSON readback.

This is useful when an assistant needs spreadsheet-style business logic without opening Excel, driving a browser, or trusting stale cached formula values.

## What it does

- Sends `sheetName`, `address`, and `value` to a Bilig WorkPaper formula endpoint.
- Confirms the endpoint reports `verified: true`.
- Returns the edited cell, recalculated values, and persistence/readback fields as compact JSON.

## Default endpoint

The default valve points at the public Bilig forecast demo:

```text
https://bilig.proompteng.ai/api/workpaper/n8n/forecast
```

The default call writes `0.5` to `Inputs!B3`. The demo recalculates expected ARR from the workbook formulas.

## Install

1. In Open WebUI, go to `Workspace -> Tools`.
2. Create a new tool.
3. Paste `main.py`.
4. Save and enable it for a model that supports tool calls.

## Example prompt

```text
Set Inputs!B3 to 0.5 and show the verified formula readback.
```

The tool should return a JSON object with `verified: true`, the edited cell, and recalculated output fields.

## Configuration

Use valves to point the tool at your own Bilig endpoint:

- `base_url`: Bilig service origin.
- `formula_path`: endpoint path that accepts a formula-readback POST.
- `timeout_seconds`: request timeout.

The endpoint must accept JSON:

```json
{
  "sheetName": "Inputs",
  "address": "B3",
  "value": 0.5
}
```

`value` can be passed as a native JSON value or as a string containing a JSON
literal. For example, `0.5`, `"0.5"`, `true`, and `"SKU-1"` are accepted.
