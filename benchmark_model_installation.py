"""V3-only verified Zenodo installation, without importing nnU-Net or Torch.

Managed receipts attest to extraction from a pinned MD5-verified archive and
record SHA-256 file hashes. They are local integrity records, not signatures.
External models are read-only and structurally checked, never assumed authentic.
The cache must be owned by the user, not writable by untrusted local users.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import ctypes
import errno
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import zipfile

from benchmark_model_registry import get_model


CHUNK_SIZE = 1024 * 1024
RECEIPT = ".v3-installation.json"


class InstallationError(RuntimeError):
    """An actionable download, validation, or installation failure."""


class _RetryableDownloadError(InstallationError):
    """Transport interruption or invalid downloaded bytes; not a disk error."""


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urlparse(new_url).scheme != "https":
            raise InstallationError("Refusing HTTPS download redirect to insecure URL")
        return super().redirect_request(request, response, code, message, headers, new_url)


@dataclass(frozen=True)
class ResolvedModel:
    model_directory: Path
    dataset_id: int
    verification: str
    provenance: dict


def default_cache_dir():
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "SpinalMultipleMyelomaSEG"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "SpinalMultipleMyelomaSEG"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "spinal-multiple-myeloma-seg"


def _digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            result.update(block)
    return result.hexdigest()


def _plain_path(path):
    """Disallow symlinks/junctions throughout cache-controlled paths."""
    path = Path(path).absolute()
    for part in (path,) + tuple(path.parents):
        reparse = part.exists() and bool(getattr(part.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if part.is_symlink() or reparse or (part.exists() and getattr(part, "is_junction", lambda: False)()):
            raise InstallationError("Refusing linked cache/model path: {}".format(part))
    return path


def _identity(spec):
    return {
        "dataset_id": spec.dataset_id, "dataset_name": spec.dataset_name,
        "zenodo_record_id": spec.zenodo_record_id, "archive_filename": spec.zip_filename,
        "archive_size": spec.zip_size, "archive_md5": spec.md5,
        "download_url": spec.download_url,
        "expected_nnunet_version": spec.expected_nnunet_version,
        "release_version": spec.release_version,
        "channels": list(spec.channel_names), "folds": list(spec.folds),
    }


def verify_archive(path, spec):
    path = _plain_path(path)
    if not path.is_file() or path.stat().st_size != spec.zip_size:
        raise InstallationError("Archive size mismatch for {} (expected {} bytes)".format(path, spec.zip_size))
    if _digest(path, "md5") != spec.md5:
        raise InstallationError("Archive MD5 mismatch for {}".format(path))


def _read_json(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise InstallationError("JSON metadata too large: {}".format(path))
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (ValueError, OSError) as error:
        raise InstallationError("Invalid metadata {}: {}".format(path, error)) from error


def validate_model(model_directory, spec):
    """Structural validation only; never loads executable checkpoint objects."""
    directory = _plain_path(model_directory)
    if directory.name != spec.trainer_directory or not directory.is_dir():
        raise InstallationError("Expected trainer directory {}: {}".format(spec.trainer_directory, directory))
    for item in directory.rglob("*"):
        _plain_path(item)
        if not item.is_dir() and not item.is_file():
            raise InstallationError("Unexpected special model file: {}".format(item))
    for filename in ("dataset.json", "plans.json"):
        if not (directory / filename).is_file():
            raise InstallationError("Missing model metadata: {}".format(directory / filename))
    dataset = _read_json(directory / "dataset.json")
    expected_channels = {str(i): name for i, name in enumerate(spec.channel_names)}
    if dataset.get("channel_names") != expected_channels:
        raise InstallationError("dataset.json channel names/order disagree with registry for model {}".format(spec.dataset_id))
    if dataset.get("file_ending") != ".nii.gz":
        raise InstallationError("dataset.json must specify .nii.gz inputs")
    plans = _read_json(directory / "plans.json")
    _, plans_name, configuration = spec.trainer_directory.split("__")
    configurations = plans.get("configurations")
    if (plans.get("dataset_name") != spec.dataset_name or plans.get("plans_name") != plans_name
            or not isinstance(configurations, dict) or not isinstance(configurations.get(configuration), dict)):
        raise InstallationError("plans.json dataset/plans/configuration disagrees with registry")
    for fold in spec.folds:
        checkpoint = directory / ("fold_{}".format(fold)) / spec.checkpoint_filename
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise InstallationError("Missing or empty checkpoint: {}".format(checkpoint))
    return directory


def _inventory(dataset_directory):
    result = {}
    for path in sorted(dataset_directory.rglob("*")):
        _plain_path(path)
        if path.is_file() and path.name != RECEIPT:
            result[path.relative_to(dataset_directory).as_posix()] = {
                "size": path.stat().st_size, "sha256": _digest(path)
            }
        elif not path.is_dir() and path.name != RECEIPT:
            raise InstallationError("Unexpected special installation file: {}".format(path))
    return result


def _safe_member(info):
    name = info.filename
    # Remove exactly one directory terminator; repeated '/' remains invalid.
    parts = (name[:-1] if info.is_dir() else name).split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {"COM{}".format(i) for i in range(1, 10)} | {"LPT{}".format(i) for i in range(1, 10)}
    if (info.orig_filename != name or not name or name.startswith("/") or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") or ":" in part or part.endswith((" ", "."))
                   or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                   or part.split(".")[0].upper() in reserved for part in parts)):
        raise InstallationError("Unsafe ZIP path: {!r}".format(name))
    mode = info.external_attr >> 16
    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise InstallationError("ZIP links/special files are forbidden: {}".format(name))
    if ((stat.S_IFMT(mode) == stat.S_IFDIR and not info.is_dir())
            or (stat.S_IFMT(mode) == stat.S_IFREG and info.is_dir())
            or (info.is_dir() and info.file_size != 0)):
        raise InstallationError("Inconsistent ZIP directory metadata: {}".format(name))
    if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise InstallationError("Encrypted/unsupported ZIP member: {}".format(name))
    return PurePosixPath(*parts)


def _rename_no_clobber(source, destination):
    """Atomically rename a complete file/directory without replacing a destination.

    os.rename is no-clobber on Windows only. Use native exclusive rename flags
    on Linux/macOS; fail closed if the OS/filesystem lacks that primitive.
    Source and destination must share a filesystem; never fall back to copying.
    """
    if os.name == "nt":
        os.rename(str(source), str(destination))
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes, destination_bytes = os.fsencode(source), os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 4)  # RENAME_EXCL
    else:
        raise InstallationError("Atomic no-clobber directory rename is unavailable on this platform")
    if result != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(code, "Installation destination already exists", str(destination))
        raise InstallationError("Atomic no-clobber installation failed: {}".format(os.strerror(code)))


def _publish_directory(source, destination):
    """Publish a complete installation using exclusive same-filesystem rename."""
    _rename_no_clobber(source, destination)


def _publish_archive(part, archive, spec):
    """Publish verified bytes without hard links, under the caller's archive lock.

    Revalidate a competing destination, never replace it. Unsupported native
    operations and cross-device renames fail closed, without download retries.
    """
    try:
        _rename_no_clobber(part, archive)
    except FileExistsError:
        verify_archive(archive, spec)


def extract_model(archive, spec, destination):
    """Verify full archive, then extract only the registry's selected subtree.

    Destination must be fresh. All archive member paths are checked, including
    unselected models; only selected model bytes are decompressed and CRC checked.
    """
    verify_archive(archive, spec)
    destination = _plain_path(destination)
    if destination.exists():
        raise InstallationError("Extraction destination already exists: {}".format(destination))
    try:
        with zipfile.ZipFile(archive) as zipped:
            selected, seen, components = [], set(), {}
            for info in zipped.infolist():
                member = _safe_member(info)
                key = str(member).casefold()
                if key in seen:
                    raise InstallationError("Duplicate/case-colliding ZIP path: {}".format(member))
                seen.add(key)
                for index in range(1, len(member.parts) + 1):
                    spelling = "/".join(member.parts[:index])
                    component_key = spelling.casefold()
                    directory = index < len(member.parts) or info.is_dir()
                    if component_key in components and components[component_key] != (spelling, directory):
                        raise InstallationError("Case-colliding or file/directory-conflicting ZIP path: {}".format(member))
                    components[component_key] = (spelling, directory)
                if info.filename.startswith(spec.archive_model_subtree):
                    relative = info.filename[len(spec.archive_root):]
                    if RECEIPT in member.parts:
                        raise InstallationError("Archive contains reserved installation receipt")
                    selected.append((info, PurePosixPath(relative)))
            if not selected:
                raise InstallationError("Expected model subtree absent: {}".format(spec.archive_model_subtree))
            expanded = sum(info.file_size for info, _ in selected)
            if expanded > spec.zip_size * 10 + 64 * 1024 * 1024:
                raise InstallationError("Excessive extracted model size")
            if shutil.disk_usage(destination.parent).free < expanded + 16 * 1024 * 1024:
                raise InstallationError("Insufficient free space for model extraction")
            destination.mkdir()
            for info, relative in selected:
                target = destination.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, CHUNK_SIZE)
                    if target.stat().st_size != info.file_size:
                        raise InstallationError("Extracted size mismatch: {}".format(relative))
            return validate_model(destination / spec.dataset_name / spec.trainer_directory, spec)
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError) as error:
        raise InstallationError("Model extraction failed: {}".format(error)) from error


class ModelInstaller:
    def __init__(self, cache_dir=None, *, opener=None, progress=None, retries=3,
                 timeout=60, backoff=1, lock_timeout=60):
        self.root = _plain_path(cache_dir if cache_dir is not None else default_cache_dir())
        self.opener = opener or urllib.request.build_opener(_HttpsRedirects()).open
        self.progress = progress or self._print_progress
        if retries < 1 or timeout <= 0 or backoff < 0 or lock_timeout < 0:
            raise ValueError("Invalid retry/timeout configuration")
        self.retries, self.timeout, self.backoff, self.lock_timeout = retries, timeout, backoff, lock_timeout

    @staticmethod
    def _print_progress(done, total):
        print("\rModel download: {}/{} bytes ({:.1%})".format(done, total, done / total), end="\n" if done == total else "", file=sys.stderr, flush=True)

    def _directory(self, *parts):
        path = _plain_path(self.root.joinpath(*map(str, parts)))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @contextmanager
    def _lock(self, name):
        path = self._directory("locks") / (name + ".lock")
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise InstallationError("Installation lock busy: {}. If a prior process crashed, verify it has stopped before removing this lock.".format(path))
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write("pid={}\n".format(os.getpid()))
            yield
        finally:
            path.unlink()

    def _download(self, spec):
        # Artifact lock is separate from model locks, so distinct lesion models
        # can install concurrently while shared archives remain serialized.
        artifact = hashlib.sha256(spec.zip_filename.encode("utf-8")).hexdigest()[:16]
        with self._lock("archive-{}-{}".format(spec.zenodo_record_id, artifact)):
            return self._download_unlocked(spec)

    def _download_unlocked(self, spec):
        archive = self._directory("archives", spec.zenodo_record_id) / spec.zip_filename
        _plain_path(archive)
        if archive.exists():
            try:
                verify_archive(archive, spec)
                return archive
            except InstallationError:
                archive.unlink()  # Only this model's invalid managed cached archive.
        if urlparse(spec.download_url).scheme != "https":
            raise InstallationError("Model downloads require HTTPS")
        if shutil.disk_usage(archive.parent).free < spec.zip_size + 16 * 1024 * 1024:
            raise InstallationError("Insufficient free space for model archive")
        part = _plain_path(archive.with_name(archive.name + ".part"))
        last_error = None
        for attempt in range(self.retries):
            try:
                # Restart from zero. No unchecked range/append semantics.
                try:
                    response = self.opener(spec.download_url, timeout=self.timeout)
                except PermissionError:
                    raise
                except (OSError, http.client.HTTPException) as error:
                    raise _RetryableDownloadError("Network request failed: {}".format(error)) from error
                with response:
                    if urlparse(response.geturl()).scheme != "https" or response.status != 200:
                        raise _RetryableDownloadError("Unexpected or insecure download response")
                    with part.open("wb") as output:
                        done = 0
                        while True:
                            try:
                                block = response.read(CHUNK_SIZE)
                            except PermissionError:
                                raise
                            except (OSError, http.client.HTTPException) as error:
                                raise _RetryableDownloadError("Network stream interrupted: {}".format(error)) from error
                            if not block:
                                break
                            done += len(block)
                            if done > spec.zip_size:
                                raise _RetryableDownloadError("Download exceeds registered archive size")
                            output.write(block)
                            self.progress(done, spec.zip_size)
                        output.flush()
                        os.fsync(output.fileno())
                try:
                    verify_archive(part, spec)
                except InstallationError as error:
                    raise _RetryableDownloadError(str(error)) from error
                _publish_archive(part, archive, spec)
                return archive
            except _RetryableDownloadError as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(self.backoff * (2 ** attempt), 30))
            except (OSError, InstallationError) as error:
                raise InstallationError("Archive filesystem/publication failure (not retried) for {}: {}".format(archive, error)) from error
            finally:
                if part.exists():
                    part.unlink()
        raise InstallationError("Download failed after {} attempts for {}: {}".format(self.retries, spec.zip_filename, last_error))

    def _managed(self, dataset, spec):
        trainer = validate_model(dataset / spec.trainer_directory, spec)
        receipt_path = _plain_path(dataset / RECEIPT)
        if not receipt_path.is_file():
            raise InstallationError("Incomplete/unverified managed installation: {}".format(dataset))
        receipt = _read_json(receipt_path)
        if (receipt.get("schema_version") != 1 or receipt.get("artifact") != _identity(spec)
                or receipt.get("verification") != "full-archive-size-md5-and-extracted-sha256"
                or receipt.get("files") != _inventory(dataset)):
            raise InstallationError("Managed model integrity/provenance mismatch: {}. Existing files were preserved.".format(dataset))
        return ResolvedModel(trainer, spec.dataset_id, "pinned-archive-and-local-sha256", receipt)

    def resolve(self, dataset_id, *, spine_model_dir=None, allow_download=True):
        """Resolve/install one model. Explicit external spine models are read-only.

        External paths must point at the trainer directory. Structural acceptance
        does NOT establish published checkpoint identity, version, or authenticity.
        """
        spec = get_model(dataset_id)
        if spine_model_dir is not None:
            if dataset_id != 802:
                raise InstallationError("External model paths are supported only for spine model 802")
            trainer = validate_model(spine_model_dir, spec)
            return ResolvedModel(trainer, dataset_id, "structural-only", {
                "source": str(trainer), "cryptographically_verified": False,
                "warning": "External checkpoint identity has not been verified against the Zenodo artifact",
            })
        with self._lock("model-{}-{}".format(spec.zenodo_record_id, spec.dataset_id)):
            parent = self._directory("models", spec.zenodo_record_id)
            dataset = _plain_path(parent / spec.dataset_name)
            if dataset.exists():
                return self._managed(dataset, spec)
            if not allow_download:
                raise InstallationError("Model {} is not installed. Re-run with --install_models to authorize installation, or supply --spine_model_dir for model 802. No download was attempted.".format(dataset_id))
            archive = self._download(spec)
            staging_root = self._directory("staging")
            with tempfile.TemporaryDirectory(prefix="model-{}-".format(dataset_id), dir=str(staging_root)) as temporary:
                staging = Path(temporary) / "extracted"
                extract_model(archive, spec, staging)
                staged_dataset = staging / spec.dataset_name
                receipt = {
                    "schema_version": 1, "artifact": _identity(spec),
                    "installed_at": datetime.now(timezone.utc).isoformat(),
                    "files": _inventory(staged_dataset),
                    "verification": "full-archive-size-md5-and-extracted-sha256",
                }
                with (staged_dataset / RECEIPT).open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, indent=2, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._managed(staged_dataset, spec)
                _publish_directory(staged_dataset, dataset)
            return self._managed(dataset, spec)

    def resolve_plan(self, plan):
        """Install only models required by a previously validated V3 execution plan."""
        resolved = {}
        if plan.spine_segmentation is None:
            resolved[802] = self.resolve(802, spine_model_dir=plan.spine_model_dir)
        if plan.lesion_model is not None:
            resolved[plan.lesion_model.dataset_id] = self.resolve(plan.lesion_model.dataset_id)
        return resolved
