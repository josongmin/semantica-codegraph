"""TypeScript Parser 테스트"""

from pathlib import Path

import pytest

from src.parser.typescript_parser import TypeScriptTreeSitterParser


@pytest.fixture
def sample_ts_file():
    """샘플 TypeScript 파일 경로"""
    return Path(__file__).parent.parent / "fixtures" / "sample_typescript.ts"


@pytest.fixture
def file_meta(sample_ts_file):
    """파일 메타데이터"""
    return {
        "repo_id": "test-repo",
        "path": "sample_typescript.ts",
        "abs_path": str(sample_ts_file),
        "language": "typescript",
    }


def test_typescript_parser_initialization():
    """TypeScript 파서 초기화 테스트"""
    parser = TypeScriptTreeSitterParser()
    assert parser is not None


def test_parse_file(file_meta):
    """파일 파싱 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, relations = parser.parse_file(file_meta)

    assert len(symbols) > 0, "심볼이 추출되어야 함"
    assert len(relations) >= 0


def test_extract_interface(file_meta):
    """인터페이스 추출 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    interfaces = [s for s in symbols if s.kind == "Interface"]
    assert len(interfaces) > 0, "User 인터페이스가 있어야 함"

    user_interface = next((s for s in symbols if s.name == "User"), None)
    assert user_interface is not None
    assert user_interface.kind == "Interface"


def test_extract_class(file_meta):
    """클래스 추출 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    classes = [s for s in symbols if s.kind == "Class"]
    assert len(classes) >= 3, "UserService, BaseRepository, UserRepository가 있어야 함"


def test_extract_abstract_class(file_meta):
    """추상 클래스 추출 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    base_repo = next((s for s in symbols if s.name == "BaseRepository"), None)
    assert base_repo is not None
    assert base_repo.attrs.get("is_abstract") is True


def test_extract_methods(file_meta):
    """메서드 추출 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    methods = [s for s in symbols if s.kind == "Method"]
    assert len(methods) > 0


def test_method_visibility(file_meta):
    """메서드 접근 제어자 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    add_user = next((s for s in symbols if "addUser" in s.name), None)
    assert add_user is not None
    assert add_user.attrs.get("visibility") == "public"


def test_static_method(file_meta):
    """정적 메서드 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    static_method = next((s for s in symbols if "createDefault" in s.name), None)
    assert static_method is not None
    assert static_method.attrs.get("is_static") is True


def test_extract_functions(file_meta):
    """함수 추출 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    functions = [s for s in symbols if s.kind == "Function"]
    assert len(functions) >= 2, "calculateAge, fetchUsers 함수가 있어야 함"


def test_async_function(file_meta):
    """비동기 함수 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    fetch_users = next((s for s in symbols if s.name == "fetchUsers"), None)
    assert fetch_users is not None
    assert fetch_users.attrs.get("is_async") is True


def test_extract_type_alias(file_meta):
    """타입 별칭 추출 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    types = [s for s in symbols if s.kind == "Type"]
    assert len(types) >= 2, "UserRole, UserWithRole 타입이 있어야 함"


def test_extends_relation(file_meta):
    """extends 관계 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, relations = parser.parse_file(file_meta)

    user_repo = next((s for s in symbols if s.name == "UserRepository"), None)
    assert user_repo is not None

    extends = user_repo.attrs.get("extends")
    assert "BaseRepository" in str(extends) if extends else False


def test_implements_relation(file_meta):
    """implements 관계 테스트"""
    parser = TypeScriptTreeSitterParser()
    symbols, _ = parser.parse_file(file_meta)

    user_repo = next((s for s in symbols if s.name == "UserRepository"), None)
    assert user_repo is not None

    implements = user_repo.attrs.get("implements", [])
    assert len(implements) > 0
