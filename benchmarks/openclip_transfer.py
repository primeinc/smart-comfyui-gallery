"""What it costs to get a prepared batch onto the GPU, and what hides it.

Separate from openclip_batch.py because it answers a different question.
That one asks how fast the encoder goes; this one asks what the
host-to-device copy costs and which configuration overlaps it with
compute.

The mechanism is upstream's, not folklore: overlapping a copy with a
kernel needs BOTH a non-default CUDA stream AND pinned source memory
(pytorch/tutorials intermediate_source/pinmem_nonblock.py:129-133, and
:225 "Only pinned tensors copies to GPU on a separate stream overlap with
another cuda kernel executed on the main stream"). `non_blocking=True`
alone removes a host-side synchronisation and nothing more -- a blocking
`.to()` issues the same `cudaMemcpyAsync` with a sync after it (:426-427).

Four sections:

    copy alone      pageable and pinned, blocking and not
    kernel alone    the encoder on a batch already resident
    naive           copy then kernel, one after the other
    double buffered copy batch N+1 while kernel N runs

Only the last can show overlap. A single copy followed immediately by a
wait has nothing to overlap WITH, which is why the first version of this
measurement showed no difference between any configuration and was
misread as "pinning does not help here".

`--trace` writes a Chrome trace through torch.profiler so the overlap can
be READ rather than inferred from a total. Wall clock says whether this
got faster; the trace says WHY, and the two are different questions.
Running it is what corrected the explanation here.

The GPU-side copy costs the same either way. What pinning removes is a
HOST-side block: a pageable copy stages through an internal pinned buffer
and holds the calling thread until the transfer is done, so the copy
stream can never issue ahead of the main one. Pinned releases the thread
and the copies start overlapping the kernels, which is what
benchmarks/results/openclip_transfer.json records under
`exposed_copy_ms_per_batch` for the double buffered pipeline: 6.49
pageable against 1.74 pinned, on a 3070 Ti at eight batches of 64. The
earlier claim that the two "hide the copy" between them described the
effect and got the mechanism wrong.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape face_pipeline_validation.py uses (:31-34).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def timed(work, repeats: int) -> float:
    """Milliseconds for `work`, the fastest of `repeats` runs.

    Synchronised on both sides: CUDA launches are asynchronous, so timing
    them without it measures how fast Python can queue work.
    """
    import torch

    work()
    torch.cuda.synchronize()
    runs = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        work()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - started) * 1000)
    return min(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="host-to-device transfer cost for a prepared CLIP batch")
    parser.add_argument("--models-dir", default=str(pathlib.Path.home() / ".smartgallery" / "models"))
    parser.add_argument("--batch", type=int, default=64, help="images per batch")
    parser.add_argument("--batches", type=int, default=8, help="batches in the pipelined sections")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--trace", default=None, help="write a Chrome trace here and profile the two pipelines")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "openclip_transfer.json"))
    asked = parser.parse_args()

    import torch

    from vision.semantic import openclip

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this measurement is about the PCIe copy")

    backend = openclip.encoder(asked.models_dir)
    device = backend.device
    shape = (asked.batch, 3, 224, 224)
    each = asked.batch * 3 * 224 * 224 * 4
    print(f"{torch.cuda.get_device_name(0)}, {backend.model_name}/{backend.checkpoint}")
    print(f"batch {asked.batch}: {shape}, {each / 1e6:.1f} MB per batch, {asked.batches} batches")

    # Shaped like a prepared batch, which is all the copy cares about.
    pageable = [torch.randn(*shape) for _ in range(asked.batches)]
    pinned = [held.pin_memory() for held in pageable]

    def infer(resident) -> None:
        with torch.no_grad():
            backend.model.encode_image(resident, normalize=True)

    print("\ncopy alone:")
    copies = {}
    for name, source, non_blocking in (
        ("pageable, blocking", pageable[0], False),
        ("pageable, non_blocking", pageable[0], True),
        ("pinned, blocking", pinned[0], False),
        ("pinned, non_blocking", pinned[0], True),
    ):
        ms = timed(lambda s=source, n=non_blocking: s.to(device, non_blocking=n), asked.repeats)
        copies[name] = round(ms, 2)
        print(f"  {name:26} {ms:7.2f} ms   {each / 1e9 / (ms / 1000):5.1f} GB/s")

    resident = pinned[0].to(device)
    kernel_ms = timed(lambda: infer(resident), asked.repeats)
    print(f"\nkernel alone, batch already on the device: {kernel_ms:.2f} ms")

    stream = torch.cuda.Stream()

    def naive(source: list) -> None:
        for held in source:
            infer(held.to(device))

    def double_buffered(source: list, *, pin: bool) -> None:
        """Copy the next batch on `stream` while the current one infers.

        `wait_stream` in both directions is what makes it correct rather
        than merely fast: the main stream waits for the copy it is about
        to read, and the copy stream waits for the kernel that is still
        reading the buffer it is about to overwrite.
        """
        main = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            first = source[0].pin_memory() if pin else source[0]
            upcoming = first.to(device, non_blocking=True)
        for at in range(len(source)):
            main.wait_stream(stream)
            current = upcoming
            if at + 1 < len(source):
                with torch.cuda.stream(stream):
                    stream.wait_stream(main)
                    nxt = source[at + 1].pin_memory() if pin else source[at + 1]
                    upcoming = nxt.to(device, non_blocking=True)
            infer(current)

    print("\npipelines:")
    pipelines = {}
    for name, work in (
        ("naive, pageable", lambda: naive(pageable)),
        ("naive, pinned", lambda: naive(pinned)),
        ("double buffered, pageable", lambda: double_buffered(pageable, pin=False)),
        ("double buffered, pinned", lambda: double_buffered(pinned, pin=False)),
        ("double buffered, pin_memory() per batch", lambda: double_buffered(pageable, pin=True)),
    ):
        ms = timed(work, max(3, asked.repeats // 2))
        pipelines[name] = round(ms, 2)
        per = ms / asked.batches
        print(
            f"  {name:40} {ms:8.2f} ms   {per:6.2f} ms/batch   {asked.batch * asked.batches / (ms / 1000):7.1f} img/sec"
        )
    exposed = {name: round(ms / asked.batches - kernel_ms, 2) for name, ms in pipelines.items()}
    print(f"\ncopy time still exposed per batch, against a {kernel_ms:.2f} ms kernel:")
    for name, ms in exposed.items():
        print(f"  {name:40} {ms:6.2f} ms")

    traced = None
    if asked.trace:
        from torch.profiler import ProfilerActivity, profile

        traced = {}
        for name, work in (
            ("double_buffered_pageable", lambda: double_buffered(pageable, pin=False)),
            ("double_buffered_pinned", lambda: double_buffered(pinned, pin=False)),
        ):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                work()
                torch.cuda.synchronize()
            where = pathlib.Path(asked.trace).with_suffix(f".{name}.json")
            where.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(where))
            traced[name] = str(where)
            print(f"\ntrace for {name}: {where}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "batch": asked.batch,
                "batches": asked.batches,
                "megabytes_per_batch": round(each / 1e6, 1),
                "copy_ms": copies,
                "kernel_ms": round(kernel_ms, 2),
                "pipeline_ms": pipelines,
                "exposed_copy_ms_per_batch": exposed,
                "traces": traced,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
