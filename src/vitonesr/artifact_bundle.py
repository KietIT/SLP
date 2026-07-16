"""Recoverable transactions for small, deterministic analysis artifact bundles.

The filesystem cannot atomically rename several independent canonical files.
This module therefore writes a hash-bound PREPARED journal before staging, then
promotes every member and writes a COMMITTED marker last.  A normal Python
exception rolls back for backwards compatibility; a process crash leaves enough
durable state for an explicit, fail-closed ``resume=True`` call.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


BUNDLE_PROTOCOL_VERSION = "analysis_artifact_bundle_v1"


@dataclass(frozen=True, slots=True)
class BundleCommit:
    destinations: tuple[Path, ...]
    provenance_path: Path
    marker_path: Path
    bundle_sha256: str


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"input changed while hashing: {source}")
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def bind_input_files(
    paths: Sequence[str | Path], *, root: str | Path | None = None
) -> tuple[dict[str, Any], ...]:
    base = Path(root) if root is not None else Path.cwd()
    resolved: dict[Path, Path] = {}
    for raw in paths:
        path = Path(raw)
        canonical = path.resolve()
        if canonical in resolved:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[canonical] = path
    bindings = []
    for canonical, path in sorted(
        resolved.items(), key=lambda item: _display_path(item[1], base).casefold()
    ):
        bindings.append(
            {
                "path": _display_path(path, base),
                "bytes": canonical.stat().st_size,
                "sha256": sha256_file(canonical),
            }
        )
    return tuple(bindings)


def verify_input_bindings(
    bindings: Sequence[Mapping[str, Any]],
    paths: Sequence[str | Path],
    *,
    root: str | Path | None = None,
) -> None:
    observed = bind_input_files(paths, root=root)
    if tuple(dict(item) for item in bindings) != observed:
        raise OSError("input artifact set changed while outputs were being computed")


def _raise(error_type: type[Exception], message: str) -> None:
    raise error_type(message)


def _write_file_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_metadata_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.write")
    try:
        _write_file_exclusive(temporary, content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(
    path: Path, *, label: str, error_type: type[Exception]
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_type(f"{label} is unreadable or corrupt: {path}") from exc
    if not isinstance(value, dict):
        _raise(error_type, f"{label} must be a JSON object: {path}")
    return value


def _marker_name(bundle_name: str) -> str:
    return f"{bundle_name}.bundle.commit.json"


def _journal_name(bundle_name: str) -> str:
    return f".{bundle_name}.bundle.transaction.json"


def _lock_name(bundle_name: str) -> str:
    return f".{bundle_name}.bundle.lock"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except (OSError, OverflowError):
        return False
    return True


def _read_lock_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
        return int(text.removeprefix("pid=")) if text.startswith("pid=") else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None


@contextmanager
def _bundle_lock(
    *,
    anchor: Path,
    bundle_name: str,
    resume: bool,
    error_type: type[Exception],
) -> Iterator[None]:
    anchor.mkdir(parents=True, exist_ok=True)
    lock_path = anchor / _lock_name(bundle_name)
    journal_path = anchor / _journal_name(bundle_name)
    marker_path = anchor / _marker_name(bundle_name)
    if (
        lock_path.exists()
        and resume
        and (journal_path.is_file() or marker_path.is_file())
    ):
        owner = _read_lock_pid(lock_path)
        if owner is not None and not _pid_is_alive(owner):
            lock_path.unlink()
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise error_type(
            f"artifact bundle is locked by another process or stale state: {lock_path}"
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _stage_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp")


def _backup_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.bak")


def _marker_backup_path(marker: Path) -> Path:
    return marker.with_name(f".{marker.name}.bak")


def _descriptor(
    *,
    bundle_name: str,
    bundle_version: str,
    destinations: Mapping[str, Path],
    contents: Mapping[str, bytes],
    inputs: Sequence[Mapping[str, Any]],
    anchor: Path,
) -> dict[str, Any]:
    outputs = [
        {
            "key": key,
            "path": _display_path(destinations[key], anchor),
            "bytes": len(contents[key]),
            "sha256": sha256_bytes(contents[key]),
        }
        for key in sorted(destinations)
    ]
    identity = {
        "protocol_version": BUNDLE_PROTOCOL_VERSION,
        "bundle_name": bundle_name,
        "bundle_version": bundle_version,
        "inputs": [dict(item) for item in inputs],
        "outputs": outputs,
    }
    return {
        **identity,
        "bundle_sha256": sha256_bytes(canonical_json_bytes(identity)),
    }


def _validate_descriptor(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    error_type: type[Exception],
) -> None:
    for key in (
        "protocol_version",
        "bundle_name",
        "bundle_version",
        "inputs",
        "outputs",
        "bundle_sha256",
    ):
        if value.get(key) != expected.get(key):
            _raise(
                error_type,
                f"{label} does not match the exact input/output bundle ({key})",
            )


def _validate_journal(
    journal: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    error_type: type[Exception],
) -> None:
    recorded = journal.get("journal_sha256")
    unsigned = {
        key: value for key, value in journal.items() if key != "journal_sha256"
    }
    if recorded != sha256_bytes(canonical_json_bytes(unsigned)):
        _raise(error_type, "bundle transaction journal integrity check failed")
    if journal.get("status") != "PREPARED" or journal.get("mode") not in {
        "create",
        "overwrite",
    }:
        _raise(error_type, "bundle transaction journal has invalid state")
    _validate_descriptor(
        journal,
        expected,
        label="bundle transaction journal",
        error_type=error_type,
    )


def _validate_marker(
    marker_path: Path,
    descriptor: Mapping[str, Any],
    destinations: Mapping[str, Path],
    *,
    anchor: Path,
    exact: bool,
    error_type: type[Exception],
) -> dict[str, Any]:
    marker = _load_json(
        marker_path, label="bundle commit marker", error_type=error_type
    )
    if marker.get("status") != "COMMITTED":
        _raise(error_type, "bundle commit marker is not COMMITTED")
    outputs = marker.get("outputs")
    if not isinstance(outputs, list):
        _raise(error_type, "bundle commit marker outputs are invalid")
    identity = {
        key: marker.get(key)
        for key in (
            "protocol_version",
            "bundle_name",
            "bundle_version",
            "inputs",
            "outputs",
        )
    }
    if marker.get("bundle_sha256") != sha256_bytes(canonical_json_bytes(identity)):
        _raise(error_type, "bundle commit marker identity is corrupt")
    if exact:
        _validate_descriptor(
            marker,
            descriptor,
            label="bundle commit marker",
            error_type=error_type,
        )
    recorded = {
        str(item.get("key")): item
        for item in outputs
        if isinstance(item, dict) and item.get("key")
    }
    if set(recorded) != set(destinations):
        _raise(error_type, "bundle commit marker output set is invalid")
    for key, destination in destinations.items():
        item = recorded[key]
        if item.get("path") != _display_path(destination, anchor):
            _raise(error_type, "bundle commit marker destination set is invalid")
        if not destination.is_file():
            _raise(error_type, f"committed bundle output is missing: {destination}")
        if destination.stat().st_size != item.get("bytes") or sha256_file(
            destination
        ) != item.get("sha256"):
            _raise(error_type, f"committed bundle output was tampered: {destination}")
    return marker


def _copy_backup(source: Path, backup: Path) -> None:
    backup.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, backup.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _clean_transaction(
    *,
    destinations: Mapping[str, Path],
    marker_path: Path,
    journal_path: Path,
) -> None:
    for destination in destinations.values():
        for auxiliary in (_stage_path(destination), _backup_path(destination)):
            if auxiliary.exists():
                auxiliary.unlink()
    marker_backup = _marker_backup_path(marker_path)
    if marker_backup.exists():
        marker_backup.unlink()
    if journal_path.exists():
        journal_path.unlink()


def _rollback(
    *,
    destinations: Mapping[str, Path],
    contents: Mapping[str, bytes],
    prior: Mapping[str, Any],
    marker_path: Path,
    prior_marker_sha256: str | None,
    error_type: type[Exception],
) -> None:
    failures: list[str] = []
    for key, destination in reversed(tuple(destinations.items())):
        prior_hash = prior.get(key)
        try:
            if prior_hash is None:
                if destination.exists() and sha256_file(destination) == sha256_bytes(
                    contents[key]
                ):
                    destination.unlink()
                continue
            backup = _backup_path(destination)
            if not backup.is_file() or sha256_file(backup) != prior_hash:
                failures.append(f"missing/tampered backup for {destination}")
                continue
            if not destination.exists() or sha256_file(destination) != prior_hash:
                backup.replace(destination)
        except OSError as exc:
            failures.append(f"restore {destination}: {exc}")
    marker_backup = _marker_backup_path(marker_path)
    try:
        if prior_marker_sha256 is None:
            if marker_path.exists():
                marker_path.unlink()
        elif marker_backup.is_file() and sha256_file(marker_backup) == prior_marker_sha256:
            marker_backup.replace(marker_path)
        else:
            failures.append("missing/tampered previous commit-marker backup")
    except OSError as exc:
        failures.append(f"restore commit marker: {exc}")
    if failures:
        _raise(
            error_type,
            "bundle commit failed and rollback was incomplete: " + "; ".join(failures),
        )


def _commit_artifact_bundle_unlocked(
    *,
    bundle_name: str,
    bundle_version: str,
    data_destinations: Mapping[str, str | Path],
    data_contents: Mapping[str, bytes],
    provenance_path: str | Path,
    input_bindings: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    resume: bool = False,
    error_type: type[Exception] = ValueError,
) -> BundleCommit:
    """Commit data files plus provenance as one recoverable logical bundle."""

    if overwrite and resume:
        _raise(error_type, "--overwrite and --resume are mutually exclusive")
    destinations = {key: Path(path) for key, path in data_destinations.items()}
    if set(destinations) != set(data_contents):
        _raise(error_type, "bundle destinations and contents use different keys")
    provenance_destination = Path(provenance_path)
    if "provenance" in destinations:
        _raise(error_type, "reserved bundle key: provenance")
    resolved = [path.resolve() for path in (*destinations.values(), provenance_destination)]
    if len(resolved) != len(set(resolved)):
        _raise(error_type, "bundle output paths must be distinct")
    anchor = provenance_destination.parent
    data_specs = [
        {
            "key": key,
            "path": _display_path(destinations[key], anchor),
            "bytes": len(data_contents[key]),
            "sha256": sha256_bytes(data_contents[key]),
        }
        for key in sorted(destinations)
    ]
    provenance = {
        "provenance_version": "analysis_artifact_provenance_v1",
        "bundle_name": bundle_name,
        "bundle_version": bundle_version,
        "inputs": [dict(item) for item in input_bindings],
        "parameters": dict(parameters or {}),
        "data_outputs": data_specs,
    }
    contents = dict(data_contents)
    contents["provenance"] = (
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    destinations["provenance"] = provenance_destination
    descriptor = _descriptor(
        bundle_name=bundle_name,
        bundle_version=bundle_version,
        destinations=destinations,
        contents=contents,
        inputs=input_bindings,
        anchor=anchor,
    )
    marker_path = anchor / _marker_name(bundle_name)
    journal_path = anchor / _journal_name(bundle_name)
    anchor.mkdir(parents=True, exist_ok=True)

    auxiliaries = [
        path
        for destination in destinations.values()
        for path in (_stage_path(destination), _backup_path(destination))
        if path.exists()
    ]
    marker_backup_existing = _marker_backup_path(marker_path)
    if marker_backup_existing.exists():
        auxiliaries.append(marker_backup_existing)

    if not journal_path.exists() and auxiliaries:
        _raise(
            error_type,
            "orphan bundle stage/backup has no transaction journal; refusing recovery: "
            + ", ".join(str(path) for path in auxiliaries),
        )

    if not resume:
        residue = list(auxiliaries)
        residue.extend(
            path
            for path in (journal_path, _marker_backup_path(marker_path))
            if path.exists()
        )
        if residue:
            _raise(
                error_type,
                "unfinished/orphan bundle transaction exists; rerun the exact command "
                "with --resume: " + ", ".join(str(path) for path in residue),
            )

    if resume and not journal_path.exists() and marker_path.exists():
        _validate_marker(
            marker_path,
            descriptor,
            destinations,
            anchor=anchor,
            exact=True,
            error_type=error_type,
        )
        return BundleCommit(
            destinations=tuple(destinations[key] for key in data_destinations),
            provenance_path=provenance_destination,
            marker_path=marker_path,
            bundle_sha256=str(descriptor["bundle_sha256"]),
        )

    occupied = [path for path in destinations.values() if path.exists()]
    if resume and not journal_path.exists() and not marker_path.exists() and occupied:
        _raise(
            error_type,
            "orphan canonical bundle has no journal/commit marker; refusing recovery: "
            + ", ".join(str(path) for path in occupied),
        )
    if not resume and not overwrite and (occupied or marker_path.exists()):
        _raise(
            error_type,
            "output already exists; use a new output directory or --overwrite: "
            + ", ".join(str(path) for path in occupied or [marker_path]),
        )

    if journal_path.exists():
        if not resume:
            _raise(
                error_type,
                f"unfinished bundle transaction exists; rerun with --resume: {journal_path}",
            )
        journal = _load_json(
            journal_path, label="bundle transaction journal", error_type=error_type
        )
        _validate_journal(journal, descriptor, error_type=error_type)
    else:
        prior: dict[str, str | None] = {}
        for key, destination in destinations.items():
            if destination.exists():
                observed = sha256_file(destination)
                expected = sha256_bytes(contents[key])
                if resume and observed != expected:
                    _raise(
                        error_type,
                        f"orphan/tampered canonical output cannot be resumed: {destination}",
                    )
                prior[key] = observed
            else:
                prior[key] = None
        prior_marker_sha256: str | None = None
        if marker_path.exists():
            if resume:
                _raise(error_type, "commit marker does not bind the requested bundle")
            _validate_marker(
                marker_path,
                descriptor,
                destinations,
                anchor=anchor,
                exact=False,
                error_type=error_type,
            )
            prior_marker_sha256 = sha256_file(marker_path)
        unsigned = {
            **descriptor,
            "status": "PREPARED",
            "mode": "overwrite" if overwrite else "create",
            "prior_sha256": prior,
            "prior_marker_sha256": prior_marker_sha256,
        }
        journal = {
            **unsigned,
            "journal_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        }
        _write_metadata_atomic(journal_path, canonical_json_bytes(journal))

    prior = journal.get("prior_sha256")
    if not isinstance(prior, dict) or set(prior) != set(destinations):
        _raise(error_type, "bundle transaction journal prior hashes are invalid")
    prior_marker_sha256 = journal.get("prior_marker_sha256")
    if prior_marker_sha256 is not None and not isinstance(prior_marker_sha256, str):
        _raise(error_type, "bundle transaction journal marker hash is invalid")

    # Make every rollback backup before the first canonical promotion.
    for key, destination in destinations.items():
        prior_hash = prior.get(key)
        backup = _backup_path(destination)
        if prior_hash is None:
            if backup.exists():
                _raise(error_type, f"orphan backup output exists: {backup}")
            continue
        if backup.exists():
            if sha256_file(backup) != prior_hash:
                _raise(error_type, f"bundle backup was tampered: {backup}")
        elif destination.is_file() and sha256_file(destination) == prior_hash:
            _copy_backup(destination, backup)
        else:
            _raise(error_type, f"required rollback backup is missing: {backup}")
    marker_backup = _marker_backup_path(marker_path)
    if prior_marker_sha256:
        if marker_backup.exists():
            if sha256_file(marker_backup) != prior_marker_sha256:
                _raise(error_type, f"commit-marker backup was tampered: {marker_backup}")
        elif marker_path.is_file() and sha256_file(marker_path) == prior_marker_sha256:
            _copy_backup(marker_path, marker_backup)
        else:
            _raise(error_type, "required commit-marker backup is missing")

    # A stage may be reconstructed only because the hash-bound journal exists.
    for key, destination in destinations.items():
        expected_hash = sha256_bytes(contents[key])
        if destination.is_file() and sha256_file(destination) == expected_hash:
            continue
        staged = _stage_path(destination)
        if staged.exists():
            if not staged.is_file() or sha256_file(staged) != expected_hash:
                _raise(error_type, f"staged bundle output was tampered: {staged}")
        else:
            _write_file_exclusive(staged, contents[key])

    if marker_path.exists():
        if prior_marker_sha256 and sha256_file(marker_path) == prior_marker_sha256:
            marker_path.unlink()
        else:
            _validate_marker(
                marker_path,
                descriptor,
                destinations,
                anchor=anchor,
                exact=True,
                error_type=error_type,
            )
            _clean_transaction(
                destinations=destinations,
                marker_path=marker_path,
                journal_path=journal_path,
            )
            return BundleCommit(
                destinations=tuple(destinations[key] for key in data_destinations),
                provenance_path=provenance_destination,
                marker_path=marker_path,
                bundle_sha256=str(descriptor["bundle_sha256"]),
            )

    try:
        for key in sorted(destinations):
            destination = destinations[key]
            expected_hash = sha256_bytes(contents[key])
            if destination.is_file():
                current = sha256_file(destination)
                if current == expected_hash:
                    continue
                if journal.get("mode") != "overwrite" or current != prior.get(key):
                    _raise(
                        error_type,
                        f"canonical output changed during bundle commit: {destination}",
                    )
            staged = _stage_path(destination)
            if not staged.is_file() or sha256_file(staged) != expected_hash:
                _raise(error_type, f"staged bundle output is missing/tampered: {staged}")
            if journal.get("mode") == "create":
                try:
                    os.link(staged, destination)
                except FileExistsError as exc:
                    raise error_type(
                        f"output appeared during no-overwrite commit: {destination}"
                    ) from exc
                staged.unlink()
            else:
                staged.replace(destination)

        marker = {**descriptor, "status": "COMMITTED"}
        _write_metadata_atomic(marker_path, canonical_json_bytes(marker))
        _validate_marker(
            marker_path,
            descriptor,
            destinations,
            anchor=anchor,
            exact=True,
            error_type=error_type,
        )
    except Exception:
        _rollback(
            destinations=destinations,
            contents=contents,
            prior=prior,
            marker_path=marker_path,
            prior_marker_sha256=prior_marker_sha256,
            error_type=error_type,
        )
        _clean_transaction(
            destinations=destinations,
            marker_path=marker_path,
            journal_path=journal_path,
        )
        raise

    _clean_transaction(
        destinations=destinations,
        marker_path=marker_path,
        journal_path=journal_path,
    )
    return BundleCommit(
        destinations=tuple(destinations[key] for key in data_destinations),
        provenance_path=provenance_destination,
        marker_path=marker_path,
        bundle_sha256=str(descriptor["bundle_sha256"]),
    )


def commit_artifact_bundle(
    *,
    bundle_name: str,
    bundle_version: str,
    data_destinations: Mapping[str, str | Path],
    data_contents: Mapping[str, bytes],
    provenance_path: str | Path,
    input_bindings: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    resume: bool = False,
    error_type: type[Exception] = ValueError,
) -> BundleCommit:
    """Serialize writers, then execute the recoverable bundle transaction."""

    anchor = Path(provenance_path).parent
    with _bundle_lock(
        anchor=anchor,
        bundle_name=bundle_name,
        resume=resume,
        error_type=error_type,
    ):
        return _commit_artifact_bundle_unlocked(
            bundle_name=bundle_name,
            bundle_version=bundle_version,
            data_destinations=data_destinations,
            data_contents=data_contents,
            provenance_path=provenance_path,
            input_bindings=input_bindings,
            parameters=parameters,
            overwrite=overwrite,
            resume=resume,
            error_type=error_type,
        )


__all__ = [
    "BUNDLE_PROTOCOL_VERSION",
    "BundleCommit",
    "bind_input_files",
    "canonical_json_bytes",
    "commit_artifact_bundle",
    "sha256_bytes",
    "sha256_file",
    "verify_input_bindings",
]
