# Consumer compatibility matrix

GENERATED. Do not edit: rebuilt from `compat/generated/*.json` by
`compat/harness/matrices.py`. Every cell traces to a case that ran.

- cases executed: **333**
- provenance: **PASS**
- consumers reproduced: **28 of 28**
- diverged: **0** / not exercised: **0**
- failing cases with no row below: **0**
- inputs skipped before a case was built: **0**

## Consumers

| consumer | commit | status | cases | necessary primitives |
| --- | --- | --- | --- | --- |
| `aligned_crop` | `--` | REPRODUCED | 5 | -- |
| `anystory` | `c38fef83a355` | REPRODUCED | 7 | -- |
| `consisid` | `c1bf18c92c62` | REPRODUCED | 4 | -- |
| `embedding_spaces` | `--` | REPRODUCED | 4 | -- |
| `face_selection` | `--` | REPRODUCED | 6 | -- |
| `gallery_storage` | `--` | REPRODUCED | 4 | -- |
| `id_lora` | `a9ab2b62dac1` | REPRODUCED | 1 | -- |
| `id_v2v` | `33dd047835cf` | REPRODUCED | 16 | -- |
| `infiniteyou` | `1c979397c5c8` | REPRODUCED | 18 | -- |
| `insightface_producer` | `--` | REPRODUCED | 6 | -- |
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
| `reactor` | `6ad6b35a4df2` | REPRODUCED | 12 | -- |
| `reference_sets` | `--` | REPRODUCED | 2 | -- |
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
| `david_normal_mp4` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `depth_mp4` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
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
| `orig_pixel_mp4` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `patch_origin` | NECESSARY AT THIS WIDTH | 0 | 0 | 50 | 4 |
| `reference_pixels` | NOT NECESSARY | 0 | 17 | 0 | 3 |
| `reference_vectors` | CHEAPER VALUE SERVES SOMETIMES | 0 | 0 | 12 | 1 |
| `selection_rule` | NECESSARY AT THIS WIDTH | 0 | 0 | 33 | 1 |
| `source_region_pixels` | NECESSARY AT THIS WIDTH | 0 | 0 | 50 | 4 |
| `source_video_bytes` | NECESSARY AT THIS WIDTH | 0 | 0 | 4 | 1 |
| `subject_mask` | NECESSARY AT THIS WIDTH | 0 | 0 | 3 | 1 |
| `whole_reference_image` | NECESSARY AT THIS WIDTH | 0 | 0 | 53 | 9 |

## Substitutions

Not necessity claims. Each asks whether a value the store ALREADY holds
can stand in for the one a consumer actually wants.

| consumer | primitive | replaced by | does it serve? |
| --- | --- | --- | --- |
| `anystory` | `subject_mask` | `face_box_mask` | **no** |
| `anystory` | `whole_reference_image` | `stored_preview` | **no** |
| `face_selection` | `selection_rule` | `other_selection_rule` | **no** |
| `id_lora` | `audio_sample_rate` | `vae_rate_assumed` | **no** |
| `id_lora` | `audio_waveform` | `mono_downmix` | **no** |
| `id_v2v` | `david_normal_mp4` | `video_round_trip` | **no** |
| `id_v2v` | `depth_mp4` | `video_round_trip` | **no** |
| `id_v2v` | `orig_pixel_mp4` | `video_round_trip` | **no** |
| `id_v2v` | `source_video_bytes` | `source_round_trip` | **no** |
| `infiniteyou` | `embedding_raw` | `stored_glintr100` | **no** |
| `infiniteyou` | `frame_dimensions` | `dimensions_swapped` | **no** |
| `infiniteyou` | `kps_source_px` | `kps_rounded_int` | **no** |
| `infiniteyou` | `patch_origin` | `origin_zero` | **no** |
| `infiniteyou` | `source_region_pixels` | `half_resolution` | **no** |
| `instantcharacter` | `whole_reference_image` | `stored_preview` | **no** |
| `instantid` | `embedding_raw` | `stored_glintr100` | **no** |
| `instantid` | `frame_dimensions` | `dimensions_swapped` | **no** |
| `instantid_upstream` | `embedding_raw` | `stored_glintr100` | **no** |
| `instantid_upstream` | `frame_dimensions` | `dimensions_swapped` | **no** |
| `ipadapter_faceid` | `embedding_raw` | `stored_glintr100` | **no** |
| `ipadapter_faceid_plus` | `embedding_raw` | `stored_glintr100` | **no** |
| `ipadapter_faceid_plus` | `kps_source_px` | `kps_rounded_int` | **no** |
| `ipadapter_faceid_plus` | `patch_origin` | `origin_zero` | **no** |
| `ipadapter_faceid_plus` | `source_region_pixels` | `half_resolution` | **no** |
| `ipadapter_upstream` | `embedding_raw` | `stored_glintr100` | **no** |
| `omnigen2` | `whole_reference_image` | `stored_preview` | **no** |
| `photomaker_v2` | `embedding_raw` | `stored_glintr100` | **no** |
| `pulid_comfyui` | `embedding_raw` | `stored_glintr100` | **no** |
| `pulid_upstream` | `embedding_raw` | `stored_glintr100` | **no** |
| `qwen_image_edit_2509` | `whole_reference_image` | `stored_preview` | **no** |
| `reactor` | `age` | `age_to_decade` | **no** |
| `reactor` | `bbox` | `through_float16` | **no** |
| `reactor` | `det_score` | `through_float16` | **no** |
| `reactor` | `embedding` | `through_float16` | **no** |
| `reactor` | `embedding_raw` | `stored_glintr100` | **no** |
| `reactor` | `gender` | `opposite_label` | **no** |
| `reactor` | `kps` | `through_float16` | **no** |
| `reactor` | `landmark_2d_106` | `through_float16` | **no** |
| `reactor` | `landmark_3d_68` | `through_float16` | **no** |
| `reference_sets` | `reference_vectors` | `order_reversed` | yes |
| `umo` | `whole_reference_image` | `stored_preview` | **no** |
| `uniportrait` | `kps_source_px` | `kps_rounded_int` | **no** |
| `uniportrait` | `patch_origin` | `origin_zero` | **no** |
| `uniportrait` | `source_region_pixels` | `half_resolution` | **no** |
| `uno` | `whole_reference_image` | `stored_preview` | **no** |
| `uso` | `whole_reference_image` | `stored_preview` | **no** |
| `xverse` | `whole_reference_image` | `stored_preview` | **no** |

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
