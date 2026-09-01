"""A producer handing over torch tensors is captured faithfully.

Run by `just probes` in its own interpreter, never by pytest: importing
torch after onnxruntime is resident crashes the DLL load on this
platform, and pytest workers run modules in arbitrary company -- so the
isolation lives in the recipe, and this child spawns nothing (sglint
SG006).
"""


def main() -> int:
    import torch

    from vision import facestore

    tensor = torch.linspace(-1, 1, 12, dtype=torch.float16).reshape(3, 4)
    blob = facestore.freeze({"feat": tensor}, producer="p", producer_version="v", container="c")
    back = facestore.thaw(blob).record["feat"]
    if not isinstance(back, torch.Tensor) or back.dtype != torch.float16:
        raise SystemExit(f"came back as {type(back).__name__}/{getattr(back, 'dtype', '?')}, not a float16 Tensor")
    if tuple(back.shape) != (3, 4) or not torch.equal(back, tensor):
        raise SystemExit("shape or values moved through the round trip")

    try:
        facestore.freeze({"x": torch.zeros(2, dtype=torch.bfloat16)}, producer="p", producer_version="v", container="c")
    except facestore.Unpreservable as why:
        if "bfloat16" not in str(why):
            raise SystemExit(f"the refusal does not name the dtype: {why}") from why
    else:
        raise SystemExit("a dtype numpy cannot spell was accepted silently")
    print("torch capture ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
