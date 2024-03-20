import pytest

from gimmecpg_python import main


@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (-2, 5, 3)])
def test_example_add(x: int, y: int, expected: int) -> None:
    assert main.add(x, y) == expected
