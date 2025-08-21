import os
import pytest
from pathlib import Path
from subprocess import run, CalledProcessError

PDF = Path("data/textbooks/sample.pdf")
ADJ = Path("data/textbooks/sample.json")

@pytest.mark.skipif(not (PDF.exists() and ADJ.exists()), reason="no sample textbook fixture")
def test_cli_runs(tmp_path):
    outdir = tmp_path / "out"
    cmd = ["python","-m","ipchat.cli","extract-textbook","--pdf",str(PDF),"--adobe-json",str(ADJ),"--out",str(outdir)]
    res = run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    outs = list(outdir.glob("*.textbook.json"))
    assert outs, res.stderr