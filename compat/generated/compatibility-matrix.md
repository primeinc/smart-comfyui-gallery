# Consumer compatibility matrix

GENERATED. Do not edit: rebuilt from `compat/generated/*.json` by
`compat/harness/matrices.py`. Every cell traces to a case that ran.

- cases executed: **302**
- provenance: **PASS**
- consumers reproduced: **22 of 22**
- diverged: **0** / not exercised: **0**
- failing cases with no row below: **0**
- inputs skipped before a case was built: **0**

## Consumers

| consumer | commit | status | cases | necessary primitives |
| --- | --- | --- | --- | --- |
| `anystory` | `c38fef83a355` | REPRODUCED | 7 | -- |
| `consisid` | `c1bf18c92c62` | REPRODUCED | 4 | -- |
| `id_lora` | `a9ab2b62dac1` | REPRODUCED | 1 | -- |
| `id_v2v` | `33dd047835cf` | REPRODUCED | 16 | -- |
| `infiniteyou` | `1c979397c5c8` | REPRODUCED | 18 | -- |
| `instantcharacter` | `5f5c49a98ba1` | REPRODUCED | 6 | -- |
| `instantid` | `72495e806bc2` | REPRODUCED | 10 | -- |
| `instantid_upstream` | `2145b67f9607` | REPRODUCED | 12 | -- |
| `ipadapter_faceid` | `a0f451a5113c` | REPRODUCED | 6 | -- |
| `ipadapter_faceid_plus` | `a0f451a5113c` | REPRODUCED | 24 | -- |
| `ipadapter_upstream` | `62e4af9d0c1a` | REPRODUCED | 5 | -- |
| `omnigen2` | `18e6f9d5271b` | REPRODUCED | 6 | -- |
| `photomaker_v2` | `060b4fcb10b7` | REPRODUCED | 6 | -- |
| `pulid_comfyui` | `93e0c4c226b8` | REPRODUCED | 6 | -- |
| `pulid_upstream` | `1aa2fc7df4bf` | REPRODUCED | 6 | -- |
| `qwen_image_edit_2509` | `6b5e1f5cec98` | REPRODUCED | 6 | -- |
| `reactor` | `6ad6b35a4df2` | REPRODUCED | 8 | -- |
| `umo` | `aada3dc32990` | REPRODUCED | 6 | -- |
| `uniportrait` | `a4deff2b48e3` | REPRODUCED | 6 | -- |
| `uno` | `a563432dcfb4` | REPRODUCED | 6 | -- |
| `uso` | `6587514aa3ad` | REPRODUCED | 6 | -- |
| `xverse` | `65e2581aac0f` | REPRODUCED | 6 | -- |

## Primitives

| primitive | verdict | breaks | survives | untested | consumers |
| --- | --- | --- | --- | --- | --- |
| `age` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `aligned_crop_112` | NECESSARY AT THIS WIDTH | 0 | 0 | 12 | 1 |
| `audio_sample_rate` | NECESSARY AT THIS WIDTH | 0 | 0 | 1 | 1 |
| `audio_waveform` | NECESSARY AT THIS WIDTH | 0 | 0 | 1 | 1 |
| `bbox` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `david_normal_mp4` | UNPROVEN | 0 | 0 | 4 | 1 |
| `depth_mp4` | UNPROVEN | 0 | 0 | 4 | 1 |
| `det_score` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `embedding` | NECESSARY AT THIS WIDTH | 0 | 0 | 10 | 2 |
| `embedding_raw` | NECESSARY AT THIS WIDTH | 0 | 0 | 56 | 10 |
| `face_rows` | NECESSARY AT THIS WIDTH | 0 | 0 | 33 | 1 |
| `frame_dimensions` | NECESSARY AT THIS WIDTH | 0 | 0 | 17 | 3 |
| `gender` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `kps` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `kps_source_px` | NECESSARY AT THIS WIDTH | 0 | 0 | 67 | 6 |
| `landmark_2d_106` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `landmark_3d_68` | NECESSARY AT THIS WIDTH | 0 | 0 | 10 | 2 |
| `orig_pixel_mp4` | UNPROVEN | 0 | 0 | 4 | 1 |
| `patch_origin` | NECESSARY AT THIS WIDTH | 0 | 0 | 50 | 4 |
| `reference_pixels` | NOT NECESSARY | 0 | 17 | 0 | 3 |
| `reference_vectors` | CHEAPER VALUE SERVES SOMETIMES | 0 | 0 | 12 | 1 |
| `selection_rule` | NECESSARY AT THIS WIDTH | 0 | 0 | 33 | 1 |
| `source_region_pixels` | NECESSARY AT THIS WIDTH | 0 | 0 | 50 | 4 |
| `source_video_bytes` | UNPROVEN | 0 | 0 | 4 | 1 |
| `subject_mask` | NECESSARY AT THIS WIDTH | 0 | 0 | 3 | 1 |
| `whole_reference_image` | NECESSARY AT THIS WIDTH | 0 | 0 | 53 | 9 |

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
| `reactor` | `gender` | `opposite_label` | **no** |
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
