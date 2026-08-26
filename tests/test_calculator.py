import asyncio

from tools.calculator import calculator


def run_calculator(expression: str) -> str:
    return asyncio.run(calculator(expression))


def test_calculator_evaluates_supported_expression():
    assert run_calculator("sqrt(16) + 2 ** 3") == "12.0"


def test_calculator_rejects_python_execution_syntax():
    result = run_calculator("__import__('os').getcwd()")

    assert result.startswith("计算错误：")
    assert "getcwd" not in result


def test_calculator_rejects_overly_complex_expression():
    result = run_calculator("1 + " * 101 + "1")

    assert result.startswith("计算错误：")