import pytest
from pydeps.pydeps import pydeps


@pytest.mark.slow
def test_no_import_cycles():
    result = pydeps(
        fname = "src/scangen/cli.py",  # Start from something that imports everything.
        only="scangen",
        no_output=True,
        show_cycles=True,
        nodot=True,  # Don't generate a dot file
        T=True,      # Don't open a browser
        return_cycles=True  # Only return cycles, don't draw graph
    )
    assert not result, f"Import cycles detected: {result}"
