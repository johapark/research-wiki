"""Private payload files and containment. Only the explicitly selected root is data."""

from contextlib import contextmanager
import codecs
import errno
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile

from ..fsatomic import exclusive_lock
from ..operations._wire import ContractError
from .contracts import BlobRef
from .errors import RecordCorrupt, RecordStoreUnavailable
from .jsonio import decode_json


def sync_directory(path):
    if os.name == "nt":
        return  # Windows has no equivalent guaranteed directory fsync here.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(fd)


def make_directory(path):
    """Make new private directories and synchronize their parent entries."""
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(cursor)
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            if not directory.is_dir():
                raise
        sync_directory(directory.parent)


@contextmanager
def storage_errors():
    try:
        yield
    except OSError as exc:
        raise RecordStoreUnavailable("archive storage unavailable; inspect the selected archive root") from exc


class Storage:
    def __init__(self, root):
        # A selected root may itself be a synced-directory symlink. Internal
        # symlinks are rejected; no blob/event chooses an arbitrary disk path.
        with storage_errors():
            self.root = Path(root).expanduser().resolve()

    def path(self, *parts):
        path = self.root.joinpath(*parts)
        if not path.is_relative_to(self.root):
            raise RecordCorrupt("archive path escapes the selected root")
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise RecordCorrupt("selected archive root changed identity")
        cursor = self.root
        for part in path.relative_to(self.root).parts:
            if part in (".", ".."):
                raise RecordCorrupt("unsafe archive reference")
            cursor /= part
            if cursor.is_symlink():
                raise RecordCorrupt("internal archive symlinks are not supported")
        return path

    def read_json(self, path):
        self.path(*path.relative_to(self.root).parts)
        if path.exists() and not path.is_file():
            raise RecordCorrupt("archive JSON record must be a regular file")
        try:
            return decode_json(path.read_bytes())
        except ContractError as exc:
            raise RecordCorrupt("invalid JSON in archive metadata/event/checkpoint") from exc

    def publish(self, path, data):
        self.path(*path.relative_to(self.root).parts)
        make_directory(path.parent)
        self.path(*path.relative_to(self.root).parts)
        fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
        staged = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, path)
            sync_directory(path.parent)
        finally:
            staged.unlink(missing_ok=True)

    def verify_blob(self, ref):
        path = self.path(*ref.relative_path.split("/"))
        if path.exists() and not path.is_file():
            raise RecordCorrupt("archive blob must be a regular file")
        hasher = hashlib.sha256()
        decoder = codecs.getincrementaldecoder('utf-8')() if ref.encoding == 'utf-8' else None
        size = 0
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    hasher.update(chunk)
                    size += len(chunk)
                    if decoder is not None:
                        decoder.decode(chunk)
                if decoder is not None:
                    decoder.decode(b'', final=True)
        except FileNotFoundError as exc:
            raise RecordCorrupt("referenced archive blob is missing") from exc
        except UnicodeError as exc:
            raise RecordCorrupt("archive text blob violates its declared encoding") from exc
        if size != ref.size_bytes or hasher.hexdigest() != ref.sha256:
            raise RecordCorrupt("archive blob digest/size mismatch")

    def put_stream(self, stream, *, media_type, encoding):
        # Validate descriptive fields even for an empty stream, before any write.
        BlobRef(sha256="0" * 64, size_bytes=0, media_type=media_type, encoding=encoding)
        staging_dir = self.path("blobs", "sha256")
        make_directory(staging_dir)
        self.path("blobs", "sha256")
        fd, name = tempfile.mkstemp(prefix=".write-", dir=staging_dir)
        staged = Path(name)
        try:
            hasher, size = hashlib.sha256(), 0
            decoder = codecs.getincrementaldecoder('utf-8')() if encoding == 'utf-8' else None
            with os.fdopen(fd, "wb") as handle:
                for chunk in iter(lambda: stream.read(65536), b""):
                    if decoder is not None:
                        decoder.decode(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
                    handle.write(chunk)
                if decoder is not None:
                    decoder.decode(b'', final=True)
                handle.flush()
                os.fsync(handle.fileno())
            ref = BlobRef(sha256=hasher.hexdigest(), size_bytes=size, media_type=media_type, encoding=encoding)
            path = self.path(*ref.relative_path.split("/"))
            with exclusive_lock(path):
                self.path(*ref.relative_path.split("/"))
                if path.exists():
                    self.verify_blob(ref)
                else:
                    make_directory(path.parent)
                    self.path(*ref.relative_path.split("/"))
                    os.replace(staged, path)
                    sync_directory(path.parent)
            return ref
        except UnicodeError as exc:
            raise ContractError("text blob is not valid UTF-8; retain undecodable content as binary") from exc
        finally:
            staged.unlink(missing_ok=True)

    def put_bytes(self, data, **kwargs):
        if type(data) is not bytes:
            raise ContractError("blob content must be bytes")
        return self.put_stream(BytesIO(data), **kwargs)
