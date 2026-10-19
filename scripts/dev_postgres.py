"""Run a command against a throwaway local PostgreSQL + pgvector server, without Docker.

Uses the ``pgserver`` wheel (bundled PostgreSQL 16 and pgvector, about 13 MB), which is
a development-only helper and not a project dependency:

    pip install pgserver
    python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres

The server's data lives in ``data/.pgdata`` (git-ignored) and the server stops when
the command finishes. The connection URL is passed to the command in
``GROUNDEDAI_TEST_POSTGRES_URL``. Add ``--keep`` to leave the data directory in place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--keep", action="store_true", help="keep data/.pgdata afterwards")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give a command after --")

    try:
        import pgserver
    except ImportError:
        print("pgserver is not installed: pip install pgserver", file=sys.stderr)
        return 2

    data_dir = REPO_ROOT / "data" / ".pgdata"
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(data_dir, cleanup_mode="stop")
    server.psql("CREATE EXTENSION IF NOT EXISTS vector;")
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    print(f"PostgreSQL running at {url}", flush=True)

    environment = dict(os.environ, GROUNDEDAI_TEST_POSTGRES_URL=url)
    # Resolve the executable ourselves: Windows CreateProcess does not apply PATHEXT.
    command = [shutil.which(command[0]) or command[0], *command[1:]]
    try:
        return subprocess.call(command, env=environment)  # noqa: S603 - developer-supplied command
    finally:
        server.cleanup()
        if not args.keep:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
