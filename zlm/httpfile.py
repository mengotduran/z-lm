"""A file-like object that reads a remote file over HTTP, fetching only the bytes it needs."""

from __future__ import annotations

import io
import time
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file whose bytes are fetched over HTTP on demand.

    A Parquet file keeps its index (the "footer") at the end, listing where every row group
    and column starts. pyarrow reads that first, then seeks straight to the bytes it needs.
    Backing those seeks with HTTP Range requests ("send me bytes 1000-1999") means we only
    download the footer and the `text` column of the row groups we actually read: a few
    hundred MB instead of the whole 2 GB shard.
    """

    def __init__(self, url: str, block: int = 4 << 20, retries: int = 5, timeout: float = 60) -> None:
        # Network trouble (a dropped Wi-Fi, switching networks) must not hang or kill a long
        # download. timeout: give up on a request that receives nothing for this many seconds
        # (without it, a dead connection waits forever). retries: then try that range again.
        self.retries, self.timeout = retries, timeout
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
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.geturl(), int(r.headers["Content-Range"].split("/")[-1])

    def _fetch(self, start: int, end: int) -> bytes:  # end inclusive, as in HTTP
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def _fetch_with_retry(self, start: int, end: int) -> bytes:
        """_fetch, retried with growing pauses (1, 2, 4, 8 s...) when the network fails.

        Retrying is always safe: each request names an exact byte range, so a repeat can't
        duplicate or skip data. A reply shorter than asked (a connection cut mid-transfer)
        counts as a failure too.
        """
        for attempt in range(self.retries + 1):
            try:
                data = self._fetch(start, end)
                if len(data) == end - start + 1:
                    self.downloaded += len(data)
                    return data
                error: Exception = OSError(f"got {len(data)} of {end - start + 1} bytes")
            except OSError as e:  # URLError, timeouts and connection resets are all OSErrors
                error = e
            if attempt < self.retries:
                wait = 2 ** attempt
                print(f"  network error ({error}), retrying in {wait}s ...", flush=True)
                time.sleep(wait)
        raise OSError(f"bytes {start}-{end}: failed after {self.retries + 1} attempts: {error}")

    def readinto(self, buffer) -> int:  # what io.RawIOBase.read() calls under the hood
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        cache_end = self.cache_start + len(self.cache)
        if not (self.cache_start <= self.pos and self.pos + n <= cache_end):
            # Not cached: fetch at least one block starting here. Large reads (a whole
            # column chunk) are fetched in one request of exactly their size.
            end = min(self.pos + max(n, self.block), self.size) - 1
            self.cache_start, self.cache = self.pos, self._fetch_with_retry(self.pos, end)
        offset = self.pos - self.cache_start
        buffer[:n] = self.cache[offset:offset + n]
        self.pos += n
        return n
