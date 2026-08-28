# Consumer compatibility matrix

GENERATED. Do not edit: rebuilt from `compat/generated/*.json` by
`compat/harness/matrices.py`. Every cell traces to a case that ran.

- cases executed: **146**
- provenance: **PASS**
- consumers reproduced: **22 of 22**
- not exercised: **0**

## Consumers

| consumer | commit | status | cases | necessary primitives |
| --- | --- | --- | --- | --- |
| `anystory` | `c38fef83a355` | REPRODUCED | 4 | whole_reference_image |
| `consisid` | `c1bf18c92c62` | REPRODUCED | 4 | whole_reference_image |
| `id_lora` | `a9ab2b62dac1` | REPRODUCED | 1 | audio_sample_rate, audio_waveform |
| `id_v2v` | `33dd047835cf` | REPRODUCED | 1 | source_video_bytes |
| `infiniteyou` | `1c979397c5c8` | REPRODUCED | 12 | embedding_raw, frame_dimensions, kps_source_px, patch_origin, source_region_pixels |
| `instantcharacter` | `5f5c49a98ba1` | REPRODUCED | 4 | whole_reference_image |
| `instantid` | `72495e806bc2` | REPRODUCED | 8 | embedding_raw, frame_dimensions, kps_source_px |
| `instantid_upstream` | `2145b67f9607` | REPRODUCED | 8 | embedding_raw, frame_dimensions, kps_source_px |
| `ipadapter_faceid` | `a0f451a5113c` | REPRODUCED | 4 | embedding_raw |
| `ipadapter_faceid_plus` | `a0f451a5113c` | REPRODUCED | 16 | embedding_raw, kps_source_px, patch_origin, source_region_pixels |
| `ipadapter_upstream` | `62e4af9d0c1a` | REPRODUCED | 4 | embedding_raw |
| `omnigen2` | `18e6f9d5271b` | REPRODUCED | 4 | whole_reference_image |
| `photomaker_v2` | `060b4fcb10b7` | REPRODUCED | 4 | embedding_raw |
| `pulid_comfyui` | `93e0c4c226b8` | REPRODUCED | 4 | embedding_raw |
| `pulid_upstream` | `1aa2fc7df4bf` | REPRODUCED | 4 | embedding_raw |
| `qwen_image_edit_2509` | `6b5e1f5cec98` | REPRODUCED | 4 | whole_reference_image |
| `reactor` | `6ad6b35a4df2` | REPRODUCED | 8 | age, bbox, det_score, embedding, embedding_raw, gender, kps, landmark_2d_106, landmark_3d_68 |
| `umo` | `aada3dc32990` | REPRODUCED | 4 | whole_reference_image |
| `uniportrait` | `a4deff2b48e3` | REPRODUCED | 4 | kps_source_px, patch_origin, source_region_pixels |
| `uno` | `a563432dcfb4` | REPRODUCED | 4 | whole_reference_image |
| `uso` | `6587514aa3ad` | REPRODUCED | 4 | whole_reference_image |
| `xverse` | `65e2581aac0f` | REPRODUCED | 4 | whole_reference_image |

## Primitives

| primitive | verdict | breaks | survives | consumers |
| --- | --- | --- | --- | --- |
| `kps_source_px` | NECESSARY | 52 | 0 | 5 |
| `embedding_raw` | NECESSARY | 40 | 0 | 10 |
| `patch_origin` | NECESSARY | 40 | 0 | 3 |
| `source_region_pixels` | NECESSARY | 40 | 0 | 3 |
| `whole_reference_image` | NECESSARY | 36 | 0 | 9 |
| `frame_dimensions` | NECESSARY | 12 | 0 | 3 |
| `embedding` | NECESSARY | 10 | 0 | 2 |
| `landmark_3d_68` | NECESSARY | 10 | 0 | 2 |
| `derive_256_from_336` | NECESSARY | 5 | 0 | 1 |
| `age` | NECESSARY | 4 | 0 | 1 |
| `bbox` | NECESSARY | 4 | 0 | 1 |
| `det_score` | NECESSARY | 4 | 0 | 1 |
| `gender` | NECESSARY | 4 | 0 | 1 |
| `kps` | NECESSARY | 4 | 0 | 1 |
| `landmark_2d_106` | NECESSARY | 4 | 0 | 1 |
| `audio_sample_rate` | NECESSARY | 1 | 0 | 1 |
| `audio_waveform` | NECESSARY | 1 | 0 | 1 |
| `source_video_bytes` | NECESSARY | 1 | 0 | 1 |
| `reference_pixels` | NOT NECESSARY | 0 | 12 | 3 |

## Substitutions

Not necessity claims. Each asks whether a value the store ALREADY holds
can stand in for the one a consumer actually wants.

| consumer | swap | does it serve? |
| --- | --- | --- |
| `anystory` | `face_patch_substituted` | **no** |
| `consisid` | `arcface_footprint_only` | **no** |
| `consisid` | `generous_patch_substituted` | **no** |
| `id_v2v` | `face_row_substituted` | **no** |
| `infiniteyou` | `stored_glintr100_substituted` | **no** |
| `instantcharacter` | `face_patch_substituted` | **no** |
| `instantid` | `stored_glintr100_substituted` | yes |
| `instantid_upstream` | `stored_glintr100_substituted` | yes |
| `ipadapter_faceid` | `stored_glintr100_substituted` | yes |
| `ipadapter_faceid_plus` | `stored_glintr100_substituted` | yes |
| `ipadapter_upstream` | `stored_glintr100_substituted` | yes |
| `omnigen2` | `face_patch_substituted` | **no** |
| `photomaker_v2` | `stored_glintr100_substituted` | **no** |
| `pulid_comfyui` | `stored_glintr100_substituted` | yes |
| `pulid_upstream` | `stored_glintr100_substituted` | yes |
| `qwen_image_edit_2509` | `face_patch_substituted` | **no** |
| `reactor` | `stored_glintr100_substituted` | yes |
| `umo` | `face_patch_substituted` | **no** |
| `uno` | `face_patch_substituted` | **no** |
| `uso` | `face_patch_substituted` | **no** |
| `xverse` | `face_patch_substituted` | **no** |

## Storage per observation

| field | dtype | shape | bytes | at 22k | at 1M |
| --- | --- | --- | --- | --- | --- |
| `embedding` | float32 | (512,) | 2,048 | 45,056,000 | 2,048,000,000 |
| `landmark_2d_106` | float32 | (106, 2) | 848 | 18,656,000 | 848,000,000 |
| `landmark_3d_68` | float32 | (68, 3) | 816 | 17,952,000 | 816,000,000 |
| `kps` | float32 | (5, 2) | 40 | 880,000 | 40,000,000 |
| `bbox` | float32 | (4,) | 16 | 352,000 | 16,000,000 |
| `pose` | float32 | (3,) | 12 | 264,000 | 12,000,000 |
| `age` | int64 | None | 8 | 176,000 | 8,000,000 |
| `det_score` | float64 | None | 8 | 176,000 | 8,000,000 |
| `gender` | int64 | None | 8 | 176,000 | 8,000,000 |
| **total** | | | **3,804** | **83,688,000** | **3,804,000,000** |
