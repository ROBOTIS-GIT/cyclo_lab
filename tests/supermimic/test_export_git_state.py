import subprocess

from cyclo_lab.utils.export_git_state import export_git_state


def test_export_diff_preserves_bytes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    git("init", "-q")
    path = repo / "notes.txt"
    path.write_bytes(b"first\n")
    git("add", "notes.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "initial")
    path.write_bytes(b"\xff\r\n")
    export_git_state(tmp_path / "run", repo)
    patch = tmp_path / "run/git/git_diff.patch"
    assert patch.read_bytes() == git("diff", "--binary", "HEAD", "--submodule=diff")
    git("apply", "--reverse", "--check", str(patch))


def test_missing_git_warns_and_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path))
    export_git_state(tmp_path / "run", tmp_path)
    assert "[WARNING]" in capsys.readouterr().out
    assert "Git state export failed" in (tmp_path / "run/git/git_state.txt").read_text()
    assert (tmp_path / "run/git/git_diff.patch").read_bytes() == b""
