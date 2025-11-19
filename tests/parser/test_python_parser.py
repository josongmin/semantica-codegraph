"""Python Parser 테스트"""

from pathlib import Path

import pytest

from src.parser.python_parser import PythonTreeSitterParser


@pytest.fixture
def sample_python_file():
    """샘플 Python 파일 경로"""
    return Path(__file__).parent.parent / "fixtures" / "sample_python.py"


@pytest.fixture
def file_meta(sample_python_file):
    """파일 메타데이터"""
    return {
        "repo_id": "test-repo",
        "path": "sample_python.py",
        "abs_path": str(sample_python_file),
        "language": "python",
    }


def test_python_parser_initialization():
    """Python 파서 초기화 테스트"""
    parser = PythonTreeSitterParser()
    assert parser is not None
    assert parser.parser is not None


def test_parse_file(file_meta):
    """파일 파싱 테스트"""
    parser = PythonTreeSitterParser()
    symbols, relations = parser.parse_file(file_meta)

    assert len(symbols) > 0, "심볼이 추출되어야 함"
    assert len(relations) >= 0, "관계가 추출될 수 있음"


def test_extract_file_symbol(file_meta):
    """File 심볼 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    file_symbol = next((s for s in symbols if s.kind == "File"), None)
    assert file_symbol is not None, "File 심볼이 있어야 함"
    assert file_symbol.name == "sample_python.py"


def test_extract_class_symbols(file_meta):
    """클래스 심볼 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    classes = [s for s in symbols if s.kind == "Class"]
    assert len(classes) == 2, "User와 Admin 클래스가 추출되어야 함"

    class_names = {c.name for c in classes}
    assert "User" in class_names
    assert "Admin" in class_names


def test_extract_user_class_details(file_meta):
    """User 클래스 상세 정보 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    user_class = next((s for s in symbols if s.name == "User"), None)
    assert user_class is not None
    assert user_class.kind == "Class"
    assert user_class.attrs.get("docstring") == "User model class"
    assert user_class.span[0] >= 0, "span이 올바르게 설정되어야 함"


def test_extract_method_symbols(file_meta):
    """메서드 심볼 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    methods = [s for s in symbols if s.kind == "Method"]
    assert len(methods) > 0, "메서드가 추출되어야 함"

    # User 클래스의 메서드 확인
    user_methods = [m for m in methods if "User." in m.name]
    assert len(user_methods) >= 3, "__init__, greet, create_default가 있어야 함"


def test_extract_greet_method_details(file_meta):
    """greet 메서드 상세 정보 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    greet_method = next((s for s in symbols if s.name == "User.greet"), None)
    assert greet_method is not None
    assert greet_method.kind == "Method"
    assert greet_method.attrs.get("parent_class") == "User"
    assert greet_method.attrs.get("docstring") == "Return greeting message"


def test_extract_static_method(file_meta):
    """정적 메서드 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    static_method = next((s for s in symbols if s.name == "User.create_default"), None)
    assert static_method is not None
    decorators = static_method.attrs.get("decorators", [])
    assert "staticmethod" in decorators, "staticmethod 데코레이터가 감지되어야 함"


def test_extract_async_method(file_meta):
    """비동기 메서드 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    async_method = next((s for s in symbols if s.name == "Admin.check_permission"), None)
    assert async_method is not None
    assert async_method.attrs.get("is_async") is True, "async 메서드로 감지되어야 함"


def test_extract_function_symbols(file_meta):
    """함수 심볼 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    functions = [s for s in symbols if s.kind == "Function"]
    assert len(functions) == 2, "calculate_total과 fetch_data 함수가 있어야 함"

    func_names = {f.name for f in functions}
    assert "calculate_total" in func_names
    assert "fetch_data" in func_names


def test_extract_async_function(file_meta):
    """비동기 함수 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    async_func = next((s for s in symbols if s.name == "fetch_data"), None)
    assert async_func is not None
    assert async_func.attrs.get("is_async") is True


def test_extract_inheritance_relation(file_meta):
    """상속 관계 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, relations = parser.parse_file(file_meta)

    # Admin이 User를 상속
    admin_class = next((s for s in symbols if s.name == "Admin"), None)
    assert admin_class is not None
    bases = admin_class.attrs.get("bases", [])
    assert "User" in bases, "Admin이 User를 상속해야 함"

    # 상속 관계 확인
    inherits_relations = [r for r in relations if r.type == "inherits"]
    assert len(inherits_relations) > 0, "상속 관계가 추출되어야 함"


def test_extract_class_method_relations(file_meta):
    """클래스-메서드 관계 추출 테스트"""
    parser = PythonTreeSitterParser()
    _, relations = parser.parse_file(file_meta)

    defines_relations = [r for r in relations if r.type == "defines"]
    assert len(defines_relations) > 0, "클래스가 메서드를 정의하는 관계가 있어야 함"


def test_span_accuracy(file_meta):
    """Span 정확도 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    for symbol in symbols:
        start_line, start_col, end_line, end_col = symbol.span
        assert start_line >= 0
        assert start_col >= 0
        assert end_line >= start_line
        if end_line == start_line:
            assert end_col >= start_col


def test_parameter_extraction(file_meta):
    """파라미터 추출 테스트"""
    parser = PythonTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    calculate_func = next((s for s in symbols if s.name == "calculate_total"), None)
    assert calculate_func is not None

    params = calculate_func.attrs.get("parameters", [])
    assert len(params) > 0, "파라미터가 추출되어야 함"
    assert any(p["name"] == "items" for p in params)
