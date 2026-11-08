"""Installer tests use tiny archives and mocked HTTPS streams, never real models."""

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

from benchmark_model_registry import get_model
from benchmark_model_installation import (
    InstallationError, ModelInstaller, RECEIPT, default_cache_dir,
    extract_model, validate_model, verify_archive, _HttpsRedirects,
    _safe_member, _publish_directory,
)


def fixture(model_id=710, extra=(), bad_channels=False, missing_fold=False):
    spec = get_model(model_id)
    stream = io.BytesIO()
    root = spec.archive_model_subtree + spec.trainer_directory + "/"
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(root + "dataset.json", json.dumps({
            "channel_names": {str(i): name for i, name in enumerate(reversed(spec.channel_names) if bad_channels else spec.channel_names)},
            "file_ending": ".nii.gz",
        }))
        archive.writestr(root + "plans.json", json.dumps({
            "dataset_name": spec.dataset_name, "plans_name": "nnUNetPlans",
            "configurations": {"3d_fullres": {}},
        }))
        for fold in spec.folds[:-1] if missing_fold else spec.folds:
            archive.writestr(root + "fold_{}/checkpoint_final.pth".format(fold), b"dummy checkpoint")
        for name, content in extra:
            if isinstance(name, str) and "\\" in name:
                # ZipInfo normalizes separators on Windows; preserve hostile bytes.
                info = zipfile.ZipInfo("placeholder")
                info.filename = info.orig_filename = name
                name = info
            archive.writestr(name, content)
    data = stream.getvalue()
    return replace(spec, zip_size=len(data), md5=hashlib.md5(data).hexdigest()), data


class Response(io.BytesIO):
    status = 200

    def geturl(self):
        return "https://zenodo.org/mock"


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def installer(self, data, **kwargs):
        return ModelInstaller(self.root / "cache", opener=lambda *a, **k: Response(data),
                              progress=lambda *a: None, backoff=0, **kwargs)

    def test_normal_explicit_directories_in_both_archive_layouts(self):
        for model_id in (802, 708, 710):
            with self.subTest(model_id=model_id):
                spec = get_model(model_id)
                root = spec.archive_model_subtree + spec.trainer_directory + "/"
                directories = [spec.archive_model_subtree, root]
                if spec.archive_root:
                    directories.insert(0, spec.archive_root)
                    directories.append(spec.archive_root + "Dataset800_other/")
                directories.extend(root + "fold_{}/".format(fold) for fold in spec.folds)
                extra = []
                for index, name in enumerate(directories):
                    info = zipfile.ZipInfo(name)
                    if index % 2:
                        info.create_system = 3
                        info.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
                    else:
                        info.create_system = 0
                        info.external_attr = 0x10  # Ordinary DOS directory entry.
                    self.assertEqual(str(_safe_member(info)), name[:-1])
                    extra.append((info, b""))
                # Directory entries deliberately follow children, as ZIP permits.
                modified, data = fixture(model_id, extra=extra)
                archive = self.root / (str(model_id) + ".zip")
                archive.write_bytes(data)
                trainer = extract_model(archive, modified, self.root / str(model_id))
                validate_model(trainer, modified)
                self.assertFalse(list((self.root / str(model_id)).rglob("Dataset800*")))

    def test_inconsistent_and_repeated_directory_entries_rejected(self):
        for name in ("directory//", "directory///", "directory//child/"):
            with self.subTest(name=name), self.assertRaisesRegex(InstallationError, "Unsafe ZIP"):
                _safe_member(zipfile.ZipInfo(name))
        info = zipfile.ZipInfo("directory/")
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        with self.assertRaisesRegex(InstallationError, "directory metadata"):
            _safe_member(info)
        info = zipfile.ZipInfo("directory/")
        info.file_size = 1
        with self.assertRaisesRegex(InstallationError, "directory metadata"):
            _safe_member(info)

    def test_atomic_publish_refuses_empty_and_nonempty_destinations(self):
        for occupied in (False, True):
            with self.subTest(occupied=occupied):
                source = self.root / ("source-" + str(occupied))
                destination = self.root / ("destination-" + str(occupied))
                source.mkdir()
                (source / "model").write_bytes(b"complete")
                destination.mkdir()
                if occupied:
                    (destination / "sentinel").write_bytes(b"preserve")
                with self.assertRaises(FileExistsError):
                    _publish_directory(source, destination)
                self.assertEqual((source / "model").read_bytes(), b"complete")
                self.assertEqual(sorted(path.name for path in destination.iterdir()), ["sentinel"] if occupied else [])
        source = self.root / "complete"
        destination = self.root / "published"
        source.mkdir()
        (source / "model").write_bytes(b"complete")
        _publish_directory(source, destination)
        self.assertFalse(source.exists())
        self.assertEqual((destination / "model").read_bytes(), b"complete")

    def test_concurrent_destination_and_publish_failure_preserve_existing_files(self):
        spec, data = fixture()
        installer = self.installer(data)
        destination = installer.root / "models" / str(spec.zenodo_record_id) / spec.dataset_name
        real_publish = _publish_directory
        def race(source, target):
            target.mkdir()
            return real_publish(source, target)
        with patch("benchmark_model_installation.get_model", return_value=spec), patch("benchmark_model_installation._publish_directory", side_effect=race):
            with self.assertRaises(FileExistsError):
                installer.resolve(710)
        self.assertTrue(destination.is_dir())
        self.assertEqual(list(destination.iterdir()), [])
        self.assertFalse(list((installer.root / "staging").iterdir()))
        self.assertFalse(list((installer.root / "locks").iterdir()))
        # A failed native publish leaves no ready installation.
        other = ModelInstaller(self.root / "other-cache", opener=lambda *a, **k: Response(data), progress=lambda *a: None)
        with patch("benchmark_model_installation.get_model", return_value=spec), patch("benchmark_model_installation._publish_directory", side_effect=InstallationError("unsupported filesystem")):
            with self.assertRaisesRegex(InstallationError, "unsupported filesystem"):
                other.resolve(710)
        self.assertFalse((other.root / "models" / str(spec.zenodo_record_id) / spec.dataset_name).exists())
        self.assertFalse(list((other.root / "staging").iterdir()))

    def test_distinct_models_in_same_record_install_concurrently(self):
        fixtures = {model: fixture(model) for model in (710, 711)}
        specs = {model: value[0] for model, value in fixtures.items()}
        payloads = {spec.download_url: fixtures[model][1] for model, spec in specs.items()}
        barrier = threading.Barrier(2)
        def opener(url, **kwargs):
            barrier.wait(timeout=5)  # Record-wide locking would fail this test.
            return Response(payloads[url])
        installers = [ModelInstaller(self.root / "parallel", opener=opener, progress=lambda *a: None, retries=1) for _ in range(2)]
        results, errors = [], []
        def work(installer, model):
            try:
                results.append(installer.resolve(model))
            except Exception as error:
                errors.append(error)
        with patch("benchmark_model_installation.get_model", side_effect=specs.__getitem__):
            threads = [threading.Thread(target=work, args=(installer, model)) for installer, model in zip(installers, specs)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(errors)
        self.assertEqual({result.dataset_id for result in results}, {710, 711})

    def test_shared_artifact_has_separate_download_lock(self):
        spec, data = fixture()
        installer = self.installer(data, lock_timeout=0)
        artifact = hashlib.sha256(spec.zip_filename.encode("utf-8")).hexdigest()[:16]
        with installer._lock("archive-{}-{}".format(spec.zenodo_record_id, artifact)):
            with self.assertRaisesRegex(InstallationError, "lock busy"):
                installer._download(spec)

    def test_native_publish_flags_and_fail_closed_paths(self):
        import errno
        from types import SimpleNamespace
        from unittest.mock import Mock
        source, destination = self.root / "source", self.root / "destination"
        for platform, function, arguments in (
            ("linux", "renameat2", (-100, os.fsencode(source), -100, os.fsencode(destination), 1)),
            ("darwin", "renamex_np", (os.fsencode(source), os.fsencode(destination), 4)),
        ):
            operation = Mock(return_value=0)
            library = SimpleNamespace(**{function: operation})
            with self.subTest(platform=platform), patch("benchmark_model_installation.os.name", "posix"), patch("benchmark_model_installation.sys.platform", platform), patch("benchmark_model_installation.ctypes.CDLL", return_value=library):
                _publish_directory(source, destination)
                operation.assert_called_once_with(*arguments)
                operation.return_value = -1
                with patch("benchmark_model_installation.ctypes.get_errno", return_value=errno.EEXIST):
                    with self.assertRaises(FileExistsError):
                        _publish_directory(source, destination)
                with patch("benchmark_model_installation.ctypes.get_errno", return_value=errno.ENOSYS):
                    with self.assertRaisesRegex(InstallationError, "Atomic no-clobber"):
                        _publish_directory(source, destination)
        with patch("benchmark_model_installation.os.name", "posix"), patch("benchmark_model_installation.sys.platform", "linux"), patch("benchmark_model_installation.ctypes.CDLL", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(InstallationError, "unavailable"):
                _publish_directory(source, destination)

    def test_install_cache_reuse_and_provenance(self):
        spec, data = fixture()
        installer = self.installer(data)
        with patch("benchmark_model_installation.get_model", return_value=spec):
            first = installer.resolve(710)
            self.assertIn("19708429", str(first.model_directory))
            self.assertEqual(first.provenance["artifact"]["archive_md5"], spec.md5)
            self.assertEqual(len(first.provenance["files"]), 7)
            with patch.object(installer, "opener", side_effect=AssertionError("network forbidden")):
                self.assertEqual(installer.resolve(710), first)
            checkpoint = first.model_directory / "fold_0/checkpoint_final.pth"
            checkpoint.write_bytes(b"corrupt")
            with self.assertRaisesRegex(InstallationError, "integrity"):
                installer.resolve(710)
            self.assertEqual(checkpoint.read_bytes(), b"corrupt")
        self.assertFalse(list((installer.root / "staging").iterdir()))
        self.assertFalse(list((installer.root / "locks").iterdir()))

    def test_spine_selective_extraction_and_external_read_only(self):
        spec, data = fixture(802, extra=(("Spinal-Multiple-Myeloma-SEG_nnUNet_models/Dataset800_other/checkpoint.pth", b"unrelated"),))
        installer = self.installer(data)
        with patch("benchmark_model_installation.get_model", return_value=spec):
            resolved = installer.resolve(802)
            self.assertTrue((resolved.model_directory / "fold_all/checkpoint_final.pth").exists())
            self.assertFalse(list(installer.root.rglob("Dataset800*")))
            before = {p: p.read_bytes() for p in resolved.model_directory.rglob("*") if p.is_file()}
            with patch.object(installer, "opener", side_effect=AssertionError("no network")):
                external = installer.resolve(802, spine_model_dir=resolved.model_directory)
            self.assertEqual(external.verification, "structural-only")
            self.assertFalse(external.provenance["cryptographically_verified"])
            self.assertEqual(before, {p: p.read_bytes() for p in resolved.model_directory.rglob("*") if p.is_file()})

    def test_cached_archive_reuse_and_bad_archive_cleanup(self):
        spec, data = fixture()
        installer = self.installer(data, retries=1)
        directory = installer._directory("archives", spec.zenodo_record_id)
        archive = directory / spec.zip_filename
        archive.write_bytes(data)
        with patch.object(installer, "opener", side_effect=AssertionError("no download")):
            self.assertEqual(installer._download(spec), archive)
        archive.write_bytes(b"bad")
        installer.opener = lambda *a, **k: Response(b"bad")
        with self.assertRaisesRegex(InstallationError, "size mismatch"):
            installer._download(spec)
        self.assertFalse(archive.exists())
        self.assertFalse(list(directory.glob("*.part")))

    def test_retry_interrupted_download_and_progress(self):
        spec, data = fixture()
        progress = []
        installer = self.installer(data)
        installer.progress = lambda done, total: progress.append((done, total))
        calls = []
        def opener(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise urllib.error.URLError("interrupted")
            return Response(data)
        installer.opener = opener
        result = installer._download(spec)
        self.assertEqual(len(calls), 2)
        self.assertIn("timeout", calls[0])
        self.assertEqual(progress[-1], (len(data), len(data)))
        verify_archive(result, spec)

    def test_midstream_failure_and_md5_retry_leave_no_partial_files(self):
        spec, data = fixture()
        class Interrupted(Response):
            def read(self, amount):
                if self.tell():
                    raise OSError("stream interrupted")
                return super().read(20)
        installer = self.installer(data, retries=2)
        attempts = [Interrupted(data), Response(data)]
        installer.opener = lambda *a, **k: attempts.pop(0)
        archive = installer._download(spec)
        self.assertEqual(archive.read_bytes(), data)
        self.assertFalse(list(archive.parent.glob("*.part")))
        archive.unlink()
        installer.opener = lambda *a, **k: Response(b"X" * len(data))
        with self.assertRaisesRegex(InstallationError, "MD5 mismatch"):
            installer._download(spec)
        self.assertFalse(archive.exists())
        self.assertFalse(list(archive.parent.glob("*.part")))

    def test_inadequate_disk_space_and_bad_plans(self):
        from types import SimpleNamespace
        spec, data = fixture()
        installer = self.installer(data)
        with patch("benchmark_model_installation.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(InstallationError, "free space"):
                installer._download(spec)
        archive = self.root / "plans.zip"
        archive.write_bytes(data)
        trainer = extract_model(archive, spec, self.root / "plans")
        (trainer / "plans.json").write_text('{"configurations": []}')
        with self.assertRaisesRegex(InstallationError, "plans.json"):
            validate_model(trainer, spec)

    def test_failed_installation_never_publishes_model(self):
        spec, data = fixture(missing_fold=True)
        installer = self.installer(data)
        with patch("benchmark_model_installation.get_model", return_value=spec):
            with self.assertRaisesRegex(InstallationError, "checkpoint"):
                installer.resolve(spec.dataset_id)
        self.assertFalse((installer.root / "models" / str(spec.zenodo_record_id) / spec.dataset_name).exists())
        self.assertFalse(list((installer.root / "staging").iterdir()))
        self.assertFalse(list((installer.root / "locks").iterdir()))

    def test_size_md5_and_missing_metadata_validation(self):
        spec, data = fixture()
        path = self.root / "archive.zip"
        path.write_bytes(data)
        with self.assertRaisesRegex(InstallationError, "MD5"):
            verify_archive(path, replace(spec, md5="0" * 32))
        for options, message in (({"bad_channels": True}, "channel"), ({"missing_fold": True}, "checkpoint")):
            broken, content = fixture(**options)
            path.write_bytes(content)
            with self.subTest(options=options), self.assertRaisesRegex(InstallationError, message):
                extract_model(path, broken, self.root / ("extract-" + message))

    def test_unsafe_zip_members(self):
        names = ("../escape", "/absolute", "C:/drive", "safe/../../escape", "safe\\escape", "safe/CON.txt", "safe/alternate:stream", "safe/trailing.")
        for index, name in enumerate(names):
            spec, data = fixture(extra=((name, b"bad"),))
            path = self.root / "unsafe.zip"
            path.write_bytes(data)
            with self.subTest(name=name), self.assertRaisesRegex(InstallationError, "Unsafe ZIP"):
                extract_model(path, spec, self.root / ("unsafe-" + str(index)))
        info = zipfile.ZipInfo("symlink")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        spec, data = fixture(extra=((info, b"target"),))
        path.write_bytes(data)
        with self.assertRaisesRegex(InstallationError, "links/special"):
            extract_model(path, spec, self.root / "linked")

    def test_collisions_and_incomplete_installations(self):
        spec, data = fixture(extra=(("other/File", b"1"), ("other/file", b"2")))
        archive = self.root / "collision.zip"
        archive.write_bytes(data)
        with self.assertRaisesRegex(InstallationError, "colliding"):
            extract_model(archive, spec, self.root / "collision")
        for index, extra in enumerate((
            (("other/File/a", b"1"), ("other/file/b", b"2")),
            (("other/file", b"1"), ("other/file/child", b"2")),
        )):
            conflict_spec, conflict_data = fixture(extra=extra)
            archive.write_bytes(conflict_data)
            with self.assertRaisesRegex(InstallationError, "conflicting|colliding"):
                extract_model(archive, conflict_spec, self.root / ("conflict-" + str(index)))
        spec, data = fixture()
        installer = self.installer(data)
        dataset = installer._directory("models", spec.zenodo_record_id, spec.dataset_name)
        sentinel = dataset / "unrelated.txt"
        sentinel.write_text("keep")
        with patch("benchmark_model_installation.get_model", return_value=spec), self.assertRaises(InstallationError):
            installer.resolve(spec.dataset_id)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_concurrent_lock_and_failure_release(self):
        installer = self.installer(b"", lock_timeout=0)
        with installer._lock("test"):
            with self.assertRaisesRegex(InstallationError, "lock busy"):
                with installer._lock("test"):
                    self.fail("lock incorrectly acquired")
        with self.assertRaises(ValueError):
            with installer._lock("test"):
                raise ValueError("failure")
        self.assertFalse(list((installer.root / "locks").iterdir()))

    def test_two_installers_share_one_download(self):
        spec, data = fixture()
        calls = []
        installer = self.installer(data)
        installer.opener = lambda *a, **k: (calls.append(1) or Response(data))
        results, failures = [], []
        def work():
            try:
                results.append(installer.resolve(710))
            except Exception as error:
                failures.append(error)
        with patch("benchmark_model_installation.get_model", return_value=spec):
            threads = [threading.Thread(target=work) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(failures)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(calls), 1)

    def test_resolution_plan_bypasses_spine_and_skips_lesions(self):
        from types import SimpleNamespace
        installer = self.installer(b"")
        supplied = SimpleNamespace(spine_segmentation=Path("mask"), lesion_model=get_model(710), spine_model_dir=None)
        with patch.object(installer, "resolve", return_value="model") as resolve:
            self.assertEqual(installer.resolve_plan(supplied), {710: "model"})
            resolve.assert_called_once_with(710)
        supplied.lesion_model = None
        with patch.object(installer, "resolve", side_effect=AssertionError("no models needed")):
            self.assertEqual(installer.resolve_plan(supplied), {})

    def test_platform_defaults_and_https_downgrade(self):
        with patch("benchmark_model_installation.sys.platform", "win32"), patch.dict(os.environ, {"LOCALAPPDATA": str(self.root)}):
            self.assertEqual(default_cache_dir(), self.root / "SpinalMultipleMyelomaSEG")
        with patch("benchmark_model_installation.sys.platform", "linux"), patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root)}):
            self.assertEqual(default_cache_dir(), self.root / "spinal-multiple-myeloma-seg")
        with self.assertRaisesRegex(InstallationError, "insecure"):
            _HttpsRedirects().redirect_request(None, None, 302, "", {}, "http://insecure.test")


if __name__ == "__main__":
    unittest.main()
