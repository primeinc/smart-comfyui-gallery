# Compatibility ledger

GENERATED from the current run by `compat/harness/ledger.py`. One row per
declared consumer, one column per stage a proof passes through.

- tree identity: `cfe487c36d68e2017e1ff33188686d2160a536167dbdef5e4980729155fcfcdd`
- declared: **28**  green: **0**  with FAILED: **0**  with BLOCKED: **28**

| consumer | src | wts | prod | emit | write | read | recon | replay | cmp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aligned_crop` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `anystory` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `consisid` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `embedding_spaces` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `face_selection` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `gallery_storage` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `id_lora` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `id_v2v` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `infiniteyou` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `insightface_producer` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `instantcharacter` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `instantid` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `instantid_upstream` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `ipadapter_faceid` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `ipadapter_faceid_plus` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `ipadapter_upstream` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `omnigen2` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `photomaker_v2` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `pulid_comfyui` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `pulid_upstream` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `qwen_image_edit_2509` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `reactor` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `reference_sets` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `umo` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `uniportrait` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `uno` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `uso` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |
| `xverse` | ok | ok | ok | ok | BLOCK | BLOCK | BLOCK | ok | ok |

## Why each non-green cell is not green

- `aligned_crop` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 25)
- `aligned_crop` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 25)
- `aligned_crop` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 25)
- `anystory` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 7)
- `anystory` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 7)
- `anystory` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 7)
- `consisid` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 4)
- `consisid` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 4)
- `consisid` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 4)
- `embedding_spaces` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 16)
- `embedding_spaces` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 16)
- `embedding_spaces` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 16)
- `face_selection` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 39)
- `face_selection` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 39)
- `face_selection` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 39)
- `gallery_storage` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 40)
- `gallery_storage` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 40)
- `gallery_storage` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 40)
- `id_lora` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 1)
- `id_lora` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 1)
- `id_lora` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 1)
- `id_v2v` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 16)
- `id_v2v` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 16)
- `id_v2v` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 16)
- `infiniteyou` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 18)
- `infiniteyou` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 18)
- `infiniteyou` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 18)
- `insightface_producer` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 18)
- `insightface_producer` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 18)
- `insightface_producer` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 18)
- `instantcharacter` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `instantcharacter` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `instantcharacter` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `instantid` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 10)
- `instantid` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 10)
- `instantid` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 10)
- `instantid_upstream` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 12)
- `instantid_upstream` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 12)
- `instantid_upstream` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 12)
- `ipadapter_faceid` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `ipadapter_faceid` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `ipadapter_faceid` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `ipadapter_faceid_plus` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 24)
- `ipadapter_faceid_plus` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 24)
- `ipadapter_faceid_plus` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 24)
- `ipadapter_upstream` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 5)
- `ipadapter_upstream` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 5)
- `ipadapter_upstream` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 5)
- `omnigen2` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `omnigen2` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `omnigen2` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `photomaker_v2` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `photomaker_v2` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `photomaker_v2` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `pulid_comfyui` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `pulid_comfyui` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `pulid_comfyui` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `pulid_upstream` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `pulid_upstream` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `pulid_upstream` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `qwen_image_edit_2509` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `qwen_image_edit_2509` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `qwen_image_edit_2509` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `reactor` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 12)
- `reactor` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 12)
- `reactor` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 12)
- `reference_sets` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 14)
- `reference_sets` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 14)
- `reference_sets` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 14)
- `umo` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `umo` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `umo` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `uniportrait` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `uniportrait` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `uniportrait` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `uno` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `uno` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `uno` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `uso` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `uso` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `uso` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
- `xverse` / durable_write: **BLOCKED** -- no durable write recorded; the runner replays from memory (durable.written_bytes absent from all 6)
- `xverse` / durable_read_back: **BLOCKED** -- no read-back recorded; nothing was re-opened (durable.read_back_sha256 absent from all 6)
- `xverse` / native_reconstruction: **BLOCKED** -- no thaw recorded; no native record was rebuilt (durable.thawed_keys absent from all 6)
