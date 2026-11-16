"""Unit tests for CSV ingestion and validation gate (FR-01, FR-02)."""

import unittest
import pandas as pd


def validate_csv_content(text: str, is_store_item: bool = True) -> tuple[bool, list[str]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False, ["File is empty or contains fewer than 2 rows."]

    header = [h.strip().lower() for h in lines[0].split(",")]
    expected_cols = (
        ["date", "store", "item", "sales"]
        if is_store_item
        else ["date", "store", "dayofweek", "sales", "customers", "open", "promo", "stateholiday", "schoolholiday"]
    )

    errors = []
    for col in expected_cols:
        if col not in header:
            errors.append(f"Missing mandatory column: '{col}'")

    if errors:
        return False, errors

    date_idx = header.index("date")
    store_idx = header.index("store")
    item_idx = header.index("item") if is_store_item else None
    sales_idx = header.index("sales")

    seen_keys = set()
    duplicate_count = 0
    invalid_dates = 0
    invalid_sales = 0

    for i in range(1, len(lines)):
        row = [r.strip() for r in lines[i].split(",")]
        if len(row) != len(header):
            errors.append(f"Row {i + 1}: Column count mismatch")
            continue

        try:
            pd.to_datetime(row[date_idx])
        except Exception:
            invalid_dates += 1

        key = (row[date_idx], row[store_idx], row[item_idx] if item_idx is not None else "")
        if key in seen_keys:
            duplicate_count += 1
        else:
            seen_keys.add(key)

        try:
            if float(row[sales_idx]) < 0:
                invalid_sales += 1
        except Exception:
            invalid_sales += 1

    if invalid_dates > 0:
        errors.append(f"Detected {invalid_dates} rows with invalid dates.")
    if duplicate_count > 0:
        errors.append(f"Detected {duplicate_count} duplicate timestamps.")
    if invalid_sales > 0:
        errors.append(f"Detected {invalid_sales} negative sales.")

    return len(errors) == 0, errors


class TestCsvValidation(unittest.TestCase):
    def test_valid_store_item_csv(self):
        valid_csv = """date,store,item,sales
2026-10-01,1,1,15
2026-10-01,1,2,22
2026-10-02,1,1,18
2026-10-02,1,2,19"""
        is_valid, errors = validate_csv_content(valid_csv, is_store_item=True)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_negative_sales(self):
        invalid_csv = """date,store,item,sales
2026-10-01,1,1,-25
2026-10-02,1,1,15"""
        is_valid, errors = validate_csv_content(invalid_csv, is_store_item=True)
        self.assertFalse(is_valid)
        self.assertTrue(any("negative sales" in e for e in errors))

    def test_missing_mandatory_column(self):
        missing_col_csv = """date,store,sales
2026-10-01,1,15"""
        is_valid, errors = validate_csv_content(missing_col_csv, is_store_item=True)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing mandatory column: 'item'" in e for e in errors))

    def test_duplicate_timestamp_rejection(self):
        duplicate_csv = """date,store,item,sales
2026-10-01,1,1,15
2026-10-01,1,1,20"""
        is_valid, errors = validate_csv_content(duplicate_csv, is_store_item=True)
        self.assertFalse(is_valid)
        self.assertTrue(any("duplicate timestamps" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
