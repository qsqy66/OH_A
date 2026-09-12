from evals.run_rag_eval import _parse_json_list, _verdict_supported


def test_parse_json_list_plain_array():
    assert _parse_json_list('["a", "b"]') == ["a", "b"]


def test_parse_json_list_strips_markdown_fence():
    assert _parse_json_list('```json\n["x"]\n```') == ["x"]


def test_parse_json_list_returns_empty_on_garbage():
    assert _parse_json_list("这不是一个 JSON 数组") == []
    assert _parse_json_list("") == []


def test_verdict_supported():
    assert _verdict_supported("支持") is True
    assert _verdict_supported("支持。") is True
    assert _verdict_supported("不支持") is False
    assert _verdict_supported("无法判断") is False
