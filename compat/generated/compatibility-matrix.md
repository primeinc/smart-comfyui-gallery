# Consumer compatibility matrix

GENERATED. Do not edit: rebuilt from `compat/generated/*.json` by
`compat/harness/matrices.py`. Every cell traces to a case that ran.

- cases executed: **299**
- provenance: **PASS**
- consumers reproduced: **17 of 22**
- diverged: **0** / partial: **5** / unsupported: **0** / not exercised: **0**
- failing cases with no row below: **19**
- inputs skipped before a case was built: **1**

## Consumers

| consumer | commit | status | cases | necessary primitives |
| --- | --- | --- | --- | --- |
| `anystory` | `c38fef83a355` | REPRODUCED | 7 | -- |
| `consisid` | `c1bf18c92c62` | REPRODUCED | 4 | -- |
| `id_lora` | `a9ab2b62dac1` | REPRODUCED | 1 | -- |
| `id_v2v` | `33dd047835cf` | PARTIAL | 12 | -- |
| `infiniteyou` | `1c979397c5c8` | REPRODUCED | 18 | -- |
| `instantcharacter` | `5f5c49a98ba1` | REPRODUCED | 6 | -- |
| `instantid` | `72495e806bc2` | REPRODUCED | 10 | -- |
| `instantid_upstream` | `2145b67f9607` | REPRODUCED | 12 | -- |
| `ipadapter_faceid` | `a0f451a5113c` | PARTIAL | 6 | -- |
| `ipadapter_faceid_plus` | `a0f451a5113c` | PARTIAL | 24 | -- |
| `ipadapter_upstream` | `62e4af9d0c1a` | PARTIAL | 6 | -- |
| `omnigen2` | `18e6f9d5271b` | REPRODUCED | 6 | -- |
| `photomaker_v2` | `060b4fcb10b7` | REPRODUCED | 6 | -- |
| `pulid_comfyui` | `93e0c4c226b8` | REPRODUCED | 6 | -- |
| `pulid_upstream` | `1aa2fc7df4bf` | REPRODUCED | 6 | -- |
| `qwen_image_edit_2509` | `6b5e1f5cec98` | REPRODUCED | 6 | -- |
| `reactor` | `6ad6b35a4df2` | REPRODUCED | 8 | -- |
| `umo` | `aada3dc32990` | REPRODUCED | 6 | -- |
| `uniportrait` | `a4deff2b48e3` | PARTIAL | 6 | -- |
| `uno` | `a563432dcfb4` | REPRODUCED | 6 | -- |
| `uso` | `6587514aa3ad` | REPRODUCED | 6 | -- |
| `xverse` | `65e2581aac0f` | REPRODUCED | 6 | -- |

## Primitives

| primitive | verdict | breaks | survives | untested | consumers |
| --- | --- | --- | --- | --- | --- |
| `age` | UNPROVEN | 0 | 0 | 4 | 1 |
| `aligned_crop_112` | UNPROVEN | 0 | 0 | 12 | 1 |
| `audio_sample_rate` | UNPROVEN | 0 | 0 | 1 | 1 |
| `audio_waveform` | UNPROVEN | 0 | 0 | 1 | 1 |
| `bbox` | UNPROVEN | 0 | 0 | 4 | 1 |
| `det_score` | UNPROVEN | 0 | 0 | 4 | 1 |
| `embedding` | UNPROVEN | 0 | 0 | 10 | 2 |
| `embedding_raw` | UNPROVEN | 0 | 0 | 54 | 10 |
| `face_rows` | UNPROVEN | 0 | 0 | 33 | 1 |
| `frame_dimensions` | UNPROVEN | 0 | 0 | 17 | 3 |
| `gender` | UNPROVEN | 0 | 0 | 4 | 1 |
| `kps` | UNPROVEN | 0 | 0 | 4 | 1 |
| `kps_source_px` | UNPROVEN | 0 | 0 | 63 | 6 |
| `landmark_2d_106` | UNPROVEN | 0 | 0 | 4 | 1 |
| `landmark_3d_68` | UNPROVEN | 0 | 0 | 10 | 2 |
| `patch_origin` | UNPROVEN | 0 | 0 | 46 | 4 |
| `reference_pixels` | NOT NECESSARY | 0 | 17 | 0 | 3 |
| `reference_vectors` | CHEAPER VALUE SERVES SOMETIMES | 0 | 0 | 12 | 1 |
| `selection_rule` | UNPROVEN | 0 | 0 | 33 | 1 |
| `source_region_pixels` | UNPROVEN | 0 | 0 | 46 | 4 |
| `source_video_bytes` | UNPROVEN | 0 | 0 | 3 | 1 |
| `subject_mask` | UNPROVEN | 0 | 0 | 3 | 1 |
| `whole_reference_image` | UNPROVEN | 0 | 0 | 53 | 9 |

## Substitutions

Not necessity claims. Each asks whether a value the store ALREADY holds
can stand in for the one a consumer actually wants.

| consumer | primitive | replaced by | does it serve? |
| --- | --- | --- | --- |
| `anystory` | `subject_mask` | `face_bbox_rectangle` | **no** |
| `anystory` | `whole_reference_image` | `face_patch` | **no** |
| `anystory` | `whole_reference_image` | `preview_derivative` | **no** |
| `consisid` | `whole_reference_image` | `arcface_footprint_only` | **no** |
| `consisid` | `whole_reference_image` | `generous_patch` | **no** |
| `id_lora` | `audio_sample_rate` | `vae_rate_assumed` | **no** |
| `id_lora` | `audio_waveform` | `pcm_16_bit` | **no** |
| `id_v2v` | `source_video_bytes` | `face_row` | **no** |
| `infiniteyou` | `embedding_raw` | `half_precision` | **no** |
| `infiniteyou` | `embedding_raw` | `stored_glintr100` | **no** |
| `infiniteyou` | `frame_dimensions` | `preview_dimensions` | **no** |
| `infiniteyou` | `kps_source_px` | `half_precision` | **no** |
| `infiniteyou` | `patch_origin` | `origin_at_zero` | **no** |
| `infiniteyou` | `source_region_pixels` | `webp_encoded` | **no** |
| `instantcharacter` | `whole_reference_image` | `face_patch` | **no** |
| `instantcharacter` | `whole_reference_image` | `preview_derivative` | **no** |
| `instantid` | `embedding_raw` | `half_precision` | **no** |
| `instantid` | `embedding_raw` | `stored_glintr100` | **no** |
| `instantid` | `frame_dimensions` | `preview_dimensions` | **no** |
| `instantid` | `kps_source_px` | `half_precision` | **no** |
| `instantid_upstream` | `embedding_raw` | `half_precision` | **no** |
| `instantid_upstream` | `embedding_raw` | `stored_glintr100` | **no** |
| `instantid_upstream` | `frame_dimensions` | `preview_dimensions` | **no** |
| `instantid_upstream` | `kps_source_px` | `half_precision` | **no** |
| `ipadapter_faceid` | `embedding_raw` | `half_precision` | **no** |
| `ipadapter_faceid` | `embedding_raw` | `stored_glintr100` | **no** |
| `ipadapter_faceid_plus` | `embedding_raw` | `half_precision` | **no** |
| `ipadapter_faceid_plus` | `embedding_raw` | `stored_glintr100` | **no** |
| `ipadapter_faceid_plus` | `kps_source_px` | `half_precision` | **no** |
| `ipadapter_faceid_plus` | `patch_origin` | `origin_at_zero` | **no** |
| `ipadapter_faceid_plus` | `source_region_pixels` | `webp_encoded` | **no** |
| `ipadapter_upstream` | `embedding_raw` | `half_precision` | **no** |
| `ipadapter_upstream` | `embedding_raw` | `stored_glintr100` | **no** |
| `omnigen2` | `whole_reference_image` | `face_patch` | **no** |
| `omnigen2` | `whole_reference_image` | `preview_derivative` | **no** |
| `photomaker_v2` | `embedding_raw` | `half_precision` | **no** |
| `photomaker_v2` | `embedding_raw` | `stored_glintr100` | **no** |
| `pulid_comfyui` | `embedding_raw` | `half_precision` | **no** |
| `pulid_comfyui` | `embedding_raw` | `stored_glintr100` | **no** |
| `pulid_upstream` | `embedding_raw` | `half_precision` | **no** |
| `pulid_upstream` | `embedding_raw` | `stored_glintr100` | **no** |
| `qwen_image_edit_2509` | `whole_reference_image` | `face_patch` | **no** |
| `qwen_image_edit_2509` | `whole_reference_image` | `preview_derivative` | **no** |
| `reactor` | `age` | `decade_bucket` | **no** |
| `reactor` | `age` | `half_precision` | **no** |
| `reactor` | `bbox` | `half_precision` | **no** |
| `reactor` | `det_score` | `half_precision` | **no** |
| `reactor` | `embedding` | `half_precision` | **no** |
| `reactor` | `embedding_raw` | `half_precision` | **no** |
| `reactor` | `embedding_raw` | `stored_glintr100` | **no** |
| `reactor` | `gender` | `half_precision` | **no** |
| `reactor` | `kps` | `half_precision` | **no** |
| `reactor` | `landmark_2d_106` | `half_precision` | **no** |
| `reactor` | `landmark_3d_68` | `half_precision` | **no** |
| `umo` | `whole_reference_image` | `face_patch` | **no** |
| `umo` | `whole_reference_image` | `preview_derivative` | **no** |
| `uniportrait` | `kps_source_px` | `half_precision` | **no** |
| `uniportrait` | `patch_origin` | `origin_at_zero` | **no** |
| `uniportrait` | `source_region_pixels` | `webp_encoded` | **no** |
| `uno` | `whole_reference_image` | `face_patch` | **no** |
| `uno` | `whole_reference_image` | `preview_derivative` | **no** |
| `uso` | `whole_reference_image` | `face_patch` | **no** |
| `uso` | `whole_reference_image` | `preview_derivative` | **no** |
| `xverse` | `whole_reference_image` | `face_patch` | **no** |
| `xverse` | `whole_reference_image` | `preview_derivative` | **no** |

## Failures with no consumer row

Lanes that are not declared consumers in the manifest, so the table
above has no row for them. Counted in the totals and gated in `main`.

| lane | cases | verdicts |
| --- | --- | --- |
| `gallery_storage` | 19 | FAIL 19 |

## Skipped inputs

Inputs a lane declined to build a case from. Recorded so the population
cannot shrink without saying so.

| lane | input | reason |
| --- | --- | --- |
| `face_selection` | `Group(asset_id='1854865030', path='C:\\ComfyUI\\output\\sample-datasets\\people_detection\\files\\medium\\1854865030.jpg', sha256='02a5a86a7f6a61dfbb02f1ed956ada43fe7144194210755116047635c186932f', bytes=509635, released_people=2, age_ranges=('60s', '60s'), description='Senior couple hikers talking in snow-covered winter nature.')` | could not be detected on: 1854865030: the detector found 1 face(s); the dataset releases 2 people. A photograph the detector does not see two faces in cannot separate `first` from `largest_bbox_area` |

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
