"""A file-like object that reads a remote file over HTTP, fetching only the bytes it needs."""

from __future__ import annotations

import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file whose bytes are fetched over HTTP on demand.

    A Parquet file keeps its index (the "footer") at the end, listing where every row group
    and column starts. pyarrow reads that first, then seeks straight to the bytes it needs.
    Backing those seeks with HTTP Range requests ("send me bytes 1000-1999") means we only
    download the footer and the `text` column of the row groups we actually read: a few
    hundred MB instead of the whole 2 GB shard.
    """

    def __init__(self, url: str, block: int = 4 << 20) -> None:
        self.url, self.size = self._probe(url)
        self.pos = 0
        self.block = block  # read ahead in 4 MB blocks: one request per block, not per read
        self.cache_start, self.cache = 0, b""
        self.downloaded = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    # The only two methods that touch the network (tests replace them with in-memory bytes).

    def _probe(self, url: str) -> tuple[str, int]:
        # Ask for byte 0 only: the reply's Content-Range header carries the total size, and
        # geturl() gives the final URL after redirects (Hugging Face redirects to a CDN).
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as r:
            return r.geturl(), int(r.headers["Content-Range"].split("/")[-1])

    def _fetch(self, start: int, end: int) -> bytes:  # end inclusive, as in HTTP
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.downloaded += len(data)
        return data

    def readinto(self, buffer) -> int:  # what io.RawIOBase.read() calls under the hood
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        cache_end = self.cache_start + len(self.cache)
        if not (self.cache_start <= self.pos and self.pos + n <= cache_end):
            # Not cached: fetch at least one block starting here. Large reads (a whole
            # column chunk) are fetched in one request of exactly their size.
            end = min(self.pos + max(n, self.block), self.size) - 1
            self.cache_start, self.cache = self.pos, self._fetch(self.pos, end)
        offset = self.pos - self.cache_start
        buffer[:n] = self.cache[offset:offset + n]
        self.pos += n
        return n
