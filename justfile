# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Full test suite in the dev venv
test:
    {{ python }} -m pytest tests/ -q

# The structural checks, on their own and in a few seconds. Each fails when
# a class of bug comes back rather than when one instance of it is noticed:
# a route added without deciding who may call it, a prompt reaching a
# visitor, an id change that leaves its ratings behind, a documented setting
# nothing reads, a setting the server honours that never reaches the page, a
# setting a layer underneath silently caps, a subprocess that can hang, a
# pipe read with no clock on it, output that cannot carry the names it
# prints, a guard stricter than the thing it guards, one writer to the
# database giving up on another sooner than the rest, a cache bounded in
# count while what it holds grows with the library, an export that drops
# what it collides with, a cache tidied only as a side effect of using it
# again, an instruction from the launcher overwritten by the thing it was
# protecting, a script the download itself rewrites so it cannot run, a
# transfer that stops early and is kept anyway, a cache with no way for its
# size to come down, a write another website can start, a name the disk
# will not keep recorded as though it did, a library discarded because it
# was reached by another name, a setting that only accepts one spelling of
# what it wants, a visitor input with no size, work done per file that no
# branch reads, a value checked for range but never for kind, a fault the
# screen cannot read, a first-run check that asks whether something is there
# rather than what it is, a search that reduces the two sides of its own
# comparison by two different rules, a day measured on a different clock
# from the one that labelled it, a route that catches its own refusal and
# reports it as a fault, a misspelt
# setting name, a
# documented way of running the suite that does not work, a password
# committed into something people download, a compose file whose variables
# do not match its own instructions, a container missing a module the app
# imports, someone's own launch script shipped in the repository, a
# password printed into a log, a route Flask registers that our own
# audit cannot see.
# `just test` runs these too; this is for running them alone.
#
# --list shows only a comment's LAST line, which turns an explanation into a
# fragment, so the summary is stated explicitly.
[doc('Structural checks alone: route gating, prompt leaks, id changes, settings, timeouts, runnability, shipped secrets')]
audit:
    {{ python }} -m pytest -q \
        tests/test_suite_is_runnable.py \
        tests/test_shipped_launchers.py \
        tests/test_compose_files.py \
        tests/test_docker_image_contents.py \
        tests/test_tracked_files.py \
        tests/test_container_entrypoint.py \
        tests/test_static_assets.py \
        tests/test_every_route_is_classified.py \
        tests/test_ai_routes_are_classified.py \
        tests/test_exhibition_leak_sweep.py \
        tests/test_file_id_changes_carry_their_data.py \
        tests/test_configuration_doc_matches_code.py \
        tests/test_comfyui_address_reaches_the_page.py \
        tests/test_upload_limit_is_the_configured_one.py \
        tests/test_console_carries_any_filename.py \
        tests/test_port_check_matches_the_server.py \
        tests/test_ai_writes_wait_for_the_scan.py \
        tests/test_view_snapshots_are_bounded_by_size.py \
        tests/test_zip_download_keeps_every_file.py \
        tests/test_prepared_downloads_expire_honestly.py \
        tests/test_shutdown_signals_respect_the_launcher.py \
        tests/test_line_endings_survive_the_clone.py \
        tests/test_download_stops_early.py \
        tests/test_thumbnail_cache_is_reclaimed.py \
        tests/test_another_site_cannot_act_on_the_gallery.py \
        tests/test_folder_rename_name_survives.py \
        tests/test_a_renamed_root_keeps_the_library.py \
        tests/test_settings_take_what_you_type.py \
        tests/test_ffmpeg_is_fetched_when_missing.py \
        tests/test_a_comment_has_a_size.py \
        tests/test_the_page_does_not_do_dead_work.py \
        tests/test_a_rating_is_a_whole_number.py \
        tests/test_an_unexpected_fault_is_still_readable.py \
        tests/test_the_library_path_must_be_a_folder.py \
        tests/test_a_model_is_found_by_its_own_name.py \
        tests/test_a_day_means_the_viewers_day.py \
        tests/test_a_missing_picture_is_said_to_be_missing.py \
        tests/test_ffprobe_reporting.py \
        tests/test_media_tool_timeouts.py \
        tests/test_video_stream_stalls.py \
        tests/test_env_var_typos.py \
        tests/test_legacy_ai_search.py

# Benchmarks through the production pipeline with live load context (bench.just)
mod bench

# Hardware matrix, decode canaries, search probes, acceptance benchmarks.
# Only the last line of this reaches --list, so it carries the summary:
# AI/ML debug surfaces (ai.just)
mod ai

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. AI_DAM_FAISS_GPU=0
# forces the fallback.
[doc('Print which faiss build the app loads, and how many GPUs it sees')]
faiss-verify:
    {{ python }} -c "from smartgallery_ai.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
