"""First-party sample inputs, per consumer, read out of each pinned clone.

Every row below was established by reading that repository's own README or
example script at the pinned commit, not by scanning for filenames. Each row
names the entrypoint that consumes the fixture and cites where upstream says
so.

A consumer with no row, or a row whose paths are absent at the pinned commit,
is VENDOR_BASELINE_UNAVAILABLE. Corpus photographs are never substituted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import proc
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Every git call here is bounded. `pinned_source.py:44` and
#: `provenance.py:44` already were, with the reason: a hang turns a red
#: gate into a run that never finishes, which reports nothing at all.

#: Where run-time-downloaded vendor inputs are cached. Beside the repository,
#: never inside it: these are third-party media under their own licences.
FETCHED: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures"


@dataclass(frozen=True)
class VendorSample:
    """One consumer's first-party example and the inputs it names."""

    consumer_id: str
    entrypoint: str
    cited: str
    inputs: tuple[str, ...]
    role: str = "single_reference"
    from_upstream: str = ""
    """Resolve `inputs` against this `[upstreams.<key>]` instead of the
    consumer's own repo. A wrapper that ships workflows but not the media they
    name has its fixture in the project it wraps."""

    urls: tuple[tuple[str, str], ...] = ()
    """(url, sha256) the upstream example downloads at run time. Cached under
    `FETCHED`, outside the repository, and verified against the recorded
    digest. Never vendored."""


#: Read from each repository at its pinned commit. `cited` is where upstream
#: states the binding.
SAMPLES: Final[tuple[VendorSample, ...]] = (
    VendorSample(
        consumer_id="instantid_upstream",
        entrypoint="infer.py",
        cited="README.md:163 load_image('./examples/yann-lecun_resize.jpg')",
        inputs=(
            "examples/yann-lecun_resize.jpg",
            "examples/musk_resize.jpeg",
            "examples/kaifu_resize.png",
            "examples/schmidhuber_resize.png",
            "examples/sam_resize.png",
        ),
    ),
    VendorSample(
        consumer_id="ipadapter_upstream",
        # The FACEID notebook, which is the boundary this consumer declares:
        # `ip_adapter/ip_adapter_faceid.py::IPAdapterFaceID.get_image_embeds`.
        # This row used to name `ip_adapter-plus-face_demo.ipynb`, which
        # constructs no FaceAnalysis at all at the pin -- it is the CLIP
        # image-encoder path -- and to carry that demo's `ai_face.png`. So a
        # detector the vendor never points at that file was being asked to
        # find a face in it, at the single 640 this consumer sweeps, and the
        # failure was recorded against our adapter. buffalo_l does find that
        # face, at 448.
        entrypoint="visualization_attnmap_faceid.ipynb",
        cited=(
            'visualization_attnmap_faceid.ipynb:72-73 FaceAnalysis(name="buffalo_l"), '
            "prepare(det_size=(640, 640)); :175 cv2.imread('assets/images/woman.png'); "
            ":177 faces[0].normed_embedding"
        ),
        inputs=("assets/images/woman.png",),
    ),
    VendorSample(
        consumer_id="photomaker_v2",
        entrypoint="examples",
        cited="examples/<identity>_<gender>/ one folder per identity",
        inputs=(
            "examples/newton_man/newton_0.jpg",
            "examples/newton_man/newton_1.jpg",
            "examples/newton_man/newton_2.png",
            "examples/newton_man/newton_3.jpg",
            "examples/scarletthead_woman/scarlett_0.jpg",
            "examples/scarletthead_woman/scarlett_1.jpg",
            "examples/scarletthead_woman/scarlett_2.jpg",
            "examples/scarletthead_woman/scarlett_3.jpg",
            "examples/lenna_woman/lenna.jpg",
        ),
        role="reference_set",
    ),
    VendorSample(
        consumer_id="consisid",
        entrypoint="infer.py",
        cited="asserts/example_images/ committed example inputs",
        inputs=tuple(f"asserts/example_images/{n}.png" for n in range(1, 6)),
    ),
    VendorSample(
        consumer_id="anystory",
        entrypoint="anystory.generate.AnyStoryFluxPipeline.generate",
        cited="README.md:42-55 images=[...] masks=[...]",
        inputs=(
            "assets/examples/1.webp",
            "assets/examples/1_mask.webp",
            "assets/examples/6_1.webp",
            "assets/examples/6_1_mask.webp",
            "assets/examples/6_2.webp",
            "assets/examples/6_2_mask.webp",
        ),
        role="masked_reference_set",
    ),
    VendorSample(
        consumer_id="id_lora",
        entrypoint="examples",
        cited="examples/reference.wav + examples/first_frame.png",
        inputs=("examples/reference.wav", "examples/first_frame.png"),
        role="audio_identity",
    ),
    VendorSample(
        consumer_id="xverse",
        entrypoint="sample",
        cited="sample/ and assets/XVerseBench/human/",
        inputs=("sample/woman.jpg", "sample/old_man.jpg", "sample/girl.jpg"),
    ),
    VendorSample(
        consumer_id="instantcharacter",
        entrypoint="assets",
        cited="assets/girl.jpg, assets/boy.jpg",
        inputs=("assets/girl.jpg", "assets/boy.jpg"),
    ),
    VendorSample(
        consumer_id="uno",
        entrypoint="assets/examples",
        cited="assets/examples/4two2one/ ref1 + ref2",
        inputs=("assets/examples/4two2one/ref1.png", "assets/examples/4two2one/ref2.png"),
        role="reference_set",
    ),
    VendorSample(
        consumer_id="umo",
        entrypoint="assets/examples",
        cited="assets/examples/UNO/4/ ref_1 + ref_2",
        inputs=("assets/examples/UNO/4/ref_1.jpg", "assets/examples/UNO/4/ref_2.png"),
        role="reference_set",
    ),
    VendorSample(
        consumer_id="instantid",
        entrypoint="examples",
        cited="README.md:53-55 image_kps input, examples/daydreaming.jpg",
        inputs=("examples/daydreaming.jpg",),
    ),
    VendorSample(
        consumer_id="reactor",
        entrypoint="comfyui-reactor-node/workflows/ReActor--Build-Blended-Face-Model--v2.json",
        cited=(
            "ComfyUI-ReActor@6ad6b35a4df2 README.md:175,:220 link this workflow. Its four "
            "LoadImage targets are not distributed anywhere: entrypoint present, no input media."
        ),
        inputs=(
            "comfyui-reactor-node/workflows/ReActor--Build-Blended-Face-Model--v1.json",
            "comfyui-reactor-node/workflows/ReActor--Build-Blended-Face-Model--v2.json",
        ),
        role="entrypoint_only",
        from_upstream="reactor_assets",
    ),
    VendorSample(
        consumer_id="qwen_image_edit_2509",
        entrypoint="QwenImageEditPlusPipeline",
        cited="Qwen-Image@6b5e1f5cec98 README.md:285-286 requests.get(...edit2509_1.jpg / edit2509_2.jpg)",
        inputs=(),
        role="reference_set",
        urls=(
            (
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg",
                "c91431b29ff9c5050e3570f3d957c179b98d42e074d780b8f3df0aca0fa5d774",
            ),
            (
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg",
                "4c081d9beaee2c775946a40839f94df9622ff6cd02ea17ea3d00f26d1a18e085",
            ),
        ),
    ),
    VendorSample(
        consumer_id="uniportrait",
        entrypoint="gradio_app.py",
        cited=(
            "README.md:49 'python gradio_app.py'; assets/examples/ tracked at a4deff2b48e3 "
            "(git ls-tree). Hidden from `rg --files` by .gitignore."
        ),
        inputs=(
            "assets/examples/1-newton.jpg",
            "assets/examples/2-stylegan2-ffhq-0100.png",
            "assets/examples/2-stylegan2-ffhq-0293.png",
            "assets/examples/3-stylegan2-ffhq-0293.png",
            "assets/examples/3-stylegan2-ffhq-0381.png",
        ),
        role="reference_set",
    ),
    VendorSample(
        consumer_id="ipadapter_faceid",
        entrypoint="examples",
        cited=(
            "ComfyUI_IPAdapter_plus@a0f451a5113c README.md:2 'reference implementation for "
            "IPAdapter models'; media in tencent-ailab/IP-Adapter"
        ),
        inputs=("assets/images/ai_face.png", "assets/images/ai_face2.png", "assets/images/woman.png"),
        from_upstream="ipadapter_upstream",
    ),
    VendorSample(
        consumer_id="ipadapter_faceid_plus",
        entrypoint="examples",
        cited="ComfyUI_IPAdapter_plus@a0f451a5113c README.md:2; media in tencent-ailab/IP-Adapter",
        inputs=("assets/images/ai_face.png", "assets/images/ai_face2.png", "assets/images/woman.png"),
        from_upstream="ipadapter_upstream",
    ),
    VendorSample(
        consumer_id="pulid_comfyui",
        entrypoint="examples",
        cited=(
            "PuLID_ComfyUI@93e0c4c226b8 README.md:3 'PuLID ComfyUI native implementation'; "
            "media in ToTheBeginning/PuLID"
        ),
        inputs=(
            "example_inputs/liuyifei.png",
            "example_inputs/lecun.jpg",
            "example_inputs/hinton.jpeg",
            "example_inputs/rihanna.webp",
        ),
        from_upstream="pulid_upstream",
    ),
    VendorSample(
        consumer_id="infiniteyou",
        entrypoint="examples/infinite_you_workflow.json",
        cited=(
            "ComfyUI_InfiniteYou@1c979397c5c8 workflow JSONs name woman.jpg and man.jpg; "
            "media lives in bytedance/InfiniteYou"
        ),
        inputs=("assets/examples/woman.jpg", "assets/examples/man.jpg", "assets/examples/man_pose.jpg"),
        from_upstream="infiniteyou_main",
    ),
    VendorSample(
        consumer_id="pulid_upstream",
        entrypoint="app.py",
        cited="example_inputs/ committed example inputs",
        inputs=(
            "example_inputs/liuyifei.png",
            "example_inputs/lifeifei.jpg",
            "example_inputs/lecun.jpg",
            "example_inputs/hinton.jpeg",
            "example_inputs/rihanna.webp",
            "example_inputs/pengwei.jpg",
            "example_inputs/zcy.webp",
        ),
    ),
    VendorSample(
        consumer_id="uso",
        entrypoint="workflow",
        cited="workflow/input.png and workflow/example*.png",
        inputs=("workflow/input.png", *(f"workflow/example{n}.png" for n in range(1, 7))),
    ),
    VendorSample(
        consumer_id="omnigen2",
        entrypoint="example_images",
        cited="example_images/ referenced by the repository's own example scripts",
        inputs=(
            "example_images/000050281.jpg",
            "example_images/000077066.jpg",
            "example_images/000119733.jpg",
            "example_images/000365954.jpg",
            "example_images/000440817.jpg",
            "example_images/01.jpg",
            "example_images/02.jpg",
            "example_images/04.jpg",
        ),
    ),
    VendorSample(
        consumer_id="id_v2v",
        entrypoint="scripts/preprocess.sh",
        cited="scripts/preprocess.sh SAMPLE_DIR default test_samples/restylization/two_sitting_woman",
        inputs=(
            "test_samples/restylization/two_sitting_woman/source.mp4",
            "test_samples/restylization/music_band/source.mp4",
            "test_samples/first_last_frame/two_women_spotlight/source.mp4",
            "test_samples/non_aligned_keyframe/suits/source.mp4",
        ),
        role="temporal_source",
    ),
)


@dataclass
class Resolved:
    """One fixture, hashed from the clone's own bytes at the pinned commit.

    `present` carried two different meanings and one name. For a committed
    blob it meant "git handed the bytes back"; for a cached URL it meant "the
    bytes match the digest the manifest declares". `conformance.py` consumed
    one boolean for both, so a cached download that failed its digest and a
    file that is simply absent were the same row. `origin` names which kind of
    row this is and `matches_expected` carries the second question separately.
    """

    consumer_id: str
    entrypoint: str
    cited: str
    role: str
    path: str
    present: bool
    """The bytes are available AND are the ones the pin or the manifest names.
    A row is usable if and only if this is true, whichever `origin` it has."""

    origin: str = "git_blob"
    """`git_blob` -- read out of the clone at the pinned commit, where the blob
    IS the expectation. `url_cache` -- fetched by hand into `FETCHED`, where
    the manifest's declared digest is the expectation."""

    matches_expected: bool | None = None
    """For a `url_cache` row, whether the cached bytes hash to the declared
    digest. None for a `git_blob` row, which has no separate expectation to
    compare against."""

    sha256: str = ""
    bytes: int = 0


def _blob(repo: Path, commit: str, path: str) -> bytes | None:
    code, out, _ = proc.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    return out if code == 0 else None


def resolve() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    pinned = {one["id"]: one for one in manifest.get("consumers", []) if one.get("repo")}
    upstreams: dict[str, Any] = manifest.get("upstreams", {})

    rows: list[Resolved] = []
    for sample in SAMPLES:
        row = pinned.get(sample.consumer_id)
        if row is None:
            continue
        holder = row
        if sample.from_upstream:
            # An `[upstreams.<key>]` or another consumer's row: a ComfyUI
            # wrapper's media lives in the project it wraps, which may be
            # pinned either way.
            holder = upstreams.get(sample.from_upstream) or pinned[sample.from_upstream]
        clone = provenance.clone_dir(refs_root, holder["repo"])
        for path in sample.inputs:
            blob = _blob(clone, holder["commit"], path)
            rows.append(
                Resolved(
                    consumer_id=sample.consumer_id,
                    entrypoint=sample.entrypoint,
                    cited=sample.cited,
                    role=sample.role,
                    path=path,
                    present=blob is not None,
                    origin="git_blob",
                    # `if blob` is False for a zero-byte committed file, so an
                    # empty blob recorded present=True with an EMPTY digest --
                    # and `conformance.py` deduplicates on the digest, so every
                    # such row collapsed into one.
                    sha256=hashlib.sha256(blob).hexdigest() if blob is not None else "",
                    bytes=len(blob) if blob is not None else 0,
                )
            )
        for url, expected in sample.urls:
            cached = FETCHED / sample.consumer_id / url.rsplit("/", 1)[-1]
            got = cached.read_bytes() if cached.is_file() else None
            # The digest is the identity, not the URL: a file replaced at the
            # same address is a different fixture and must not pass.
            actual = hashlib.sha256(got).hexdigest() if got else ""
            rows.append(
                Resolved(
                    consumer_id=sample.consumer_id,
                    entrypoint=sample.entrypoint,
                    cited=f"{sample.cited} -> {url}",
                    role=sample.role,
                    path=str(cached),
                    present=actual == expected,
                    origin="url_cache",
                    matches_expected=actual == expected,
                    sha256=actual,
                    bytes=len(got) if got is not None else 0,
                )
            )

    # `entrypoint_only` resolved a runnable example and NO input media. It is
    # not vendor-fixture coverage: there is nothing to feed the entrypoint, so
    # adapter conformance on vendor data cannot be established.
    with_media = {one.consumer_id for one in rows if one.present and one.role != "entrypoint_only"}
    entrypoint_only = {one.consumer_id for one in rows if one.present and one.role == "entrypoint_only"} - with_media
    declared = set(pinned)
    return {
        "runtime": provenance.runtime_identity(),
        "fixtures": [asdict(one) for one in rows],
        "population": {
            "declared": sorted(declared),
            "with_vendor_input_media": sorted(with_media),
            "entrypoint_only": sorted(entrypoint_only),
            "VENDOR_BASELINE_UNAVAILABLE": sorted(declared - with_media - entrypoint_only),
        },
    }


def main() -> int:
    out = resolve()
    missing = [one for one in out["fixtures"] if not one["present"]]
    for one in out["fixtures"]:
        if one["present"]:
            print(f"ok  {one['consumer_id']:<20} {one['path']:<58} {one['sha256'][:16]} {one['bytes']:>10,}")
    for one in missing:
        print(f"!!  {one['consumer_id']:<20} {one['path']:<58} ABSENT at pinned commit")

    pop = out["population"]
    print(f"\nfixtures resolved            : {sum(1 for one in out['fixtures'] if one['present'])}")
    print(f"with vendor INPUT MEDIA      : {len(pop['with_vendor_input_media'])} of {len(pop['declared'])}")
    print(f"entrypoint only, no input    : {len(pop['entrypoint_only'])}  {pop['entrypoint_only']}")
    unresolved = pop["VENDOR_BASELINE_UNAVAILABLE"]
    print(f"VENDOR_BASELINE_UNAVAILABLE  : {len(unresolved)}  {unresolved}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "vendor_fixtures.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
