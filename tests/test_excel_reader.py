import asyncio

from tools.excel_reader import read_excel


def run_read_excel(filename: str, sheet_name: str) -> str:
    return asyncio.run(read_excel(filename, sheet_name))


def test_excel_reader_rejects_path_traversal():
    result = run_read_excel("../outside.xlsx", "Sheet1")

    assert "data 目录" in result


def test_excel_reader_rejects_unsupported_extension():
    result = run_read_excel("employees.csv", "Sheet1")

    assert ".xlsx" in result and ".xls" in result