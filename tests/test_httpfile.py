import io

import pytest

from zlm.httpfile import HTTPRangeFile

DATA = bytes(range(256)) * 40  # 10,240 bytes, every offset has a recognizable value


class FakeRemote(HTTPRangeFile):
    """HTTPRangeFile with the network replaced by DATA in memory; records each request."""

    def __init__(self, block: int = 1000) -> None:
        self.requests: list[tuple[int, int]] = []
        super().__init__("fake://file", block=block)

    def _probe(self, url):
        return url, len(DATA)

    def _fetch(self, start, end):
        self.requests.append((start, end))
        self.downloaded += end - start + 1
        return DATA[start:end + 1]  # HTTP ranges include the end byte


def test_reads_match_the_file():
    f = FakeRemote()
    assert f.read(10) == DATA[:10]
    assert f.read(5) == DATA[10:15]  # continues where the last read stopped
    f.seek(5000)
    assert f.read(3000) == DATA[5000:8000]  # larger than one block


def test_seek_whence_and_tell():
    f = FakeRemote()
    f.seek(-8, io.SEEK_END)  # how Parquet readers find the footer
    assert f.tell() == len(DATA) - 8
    assert f.read() == DATA[-8:]
    f.seek(100)
    f.seek(20, io.SEEK_CUR)
    assert f.tell() == 120


def test_reading_past_the_end_stops_at_the_end():
    f = FakeRemote()
    f.seek(len(DATA) - 4)
    assert f.read(100) == DATA[-4:]
    assert f.read(100) == b""


def test_small_reads_are_served_from_one_block():
    # Many small reads within one block cost a single request: that's the read-ahead.
    f = FakeRemote(block=1000)
    for _ in range(50):
        f.read(10)
    assert f.requests == [(0, 999)]


def test_only_requested_regions_are_downloaded():
    f = FakeRemote(block=100)
    f.seek(9000)
    f.read(50)
    assert f.downloaded == 100  # one block, not the 9 KB before it


@pytest.mark.parametrize("block", [1, 7, 4096])
def test_any_block_size_reads_correctly(block):
    f = FakeRemote(block=block)
    f.seek(123)
    assert f.read(2000) == DATA[123:2123]
