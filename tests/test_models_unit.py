"""Model-free unit tests for smartgallery_ai.models.

Covers the parts that are ours rather than the model's: the vision-token
budget translated from the processor's declared geometry, the KV-cached
turn mechanics (image processed once, later turns carrying the whole id
sequence rather than only the delta), the fixed-tools contract, weights
location resolution, device/attention resolution, and the JSON/tool-call
extractors.

No weights, no torch, no transformers import: the Chat under test is built
with object.__new__ and handed a scripted processor, model and torch
stand-in, the same way the review tests build a bare reviewer.
The counterpart tests against real weights are in tests/test_real_backends.py.
"""

import contextlib
import logging
import types

import pytest
from PIL import Image

from smartgallery_ai import models as ai_models

# --- fixtures / helpers -----------------------------------------------------


class Ids:
    """A token-id batch of one row, standing in for a torch tensor.

    Supports only what Chat uses: `.shape` and `[:, n:]` slicing. Values
    stay plain ints so assertions read as lists.
    """

    def __init__(self, row):
        self.row = list(row)

    @property
    def shape(self):
        return (1, len(self.row))

    def __getitem__(self, key):
        rows, columns = key if isinstance(key, tuple) else (key, slice(None))
        row = self.row[columns]
        return Ids(row) if isinstance(rows, slice) else row

    def to(self, device):
        self.device = device
        return self

    def __eq__(self, other):
        return isinstance(other, Ids) and self.row == other.row

    def __repr__(self):
        return f"Ids({self.row})"


class FakeTorch:
    """`cat` over Ids, plus a no-op no_grad."""

    @staticmethod
    def cat(parts, dim=-1):
        del dim
        row = []
        for part in parts:
            row.extend(part.row)
        return Ids(row)

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()


class Batch(dict):
    """Stands in for BatchFeature: a dict that can be moved to a device."""

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    """Records every render, processor call and tokenizer call.

    `apply_chat_template` renders each message as `<role:body>` with one
    `<img>` per image part, so the prefix relationship between turns is
    visible in the recorded strings. Tokenizing maps characters to their
    ordinals, so id sequences are directly checkable.
    """

    def __init__(self, geometry=True):
        self.renders = []
        self.processor_calls = []
        self.tokenizer_calls = []
        self.image_processor = types.SimpleNamespace()
        if geometry:
            self.image_processor.patch_size = 16
            self.image_processor.merge_size = 2
            self.image_processor.size = types.SimpleNamespace(
                longest_edge=16777216, shortest_edge=65536)
        self.tokenizer = self._tokenize

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, tools=None):
        assert tokenize is False, "Chat must render text here, never tokenize"
        parts = []
        if tools:
            parts.append("<tools:" + ",".join(
                tool["function"]["name"] for tool in tools) + ">")
        for message in messages:
            body = ""
            for item in message["content"]:
                body += "<img>" if item["type"] == "image" else item["text"]
            parts.append(f"<{message['role']}:{body}>")
        if add_generation_prompt:
            parts.append("<gen>")
        text = "".join(parts)
        self.renders.append({"text": text, "tools": tools,
                             "generation_prompt": add_generation_prompt})
        return text

    def __call__(self, text="", images=None, return_tensors=None, **kwargs):
        del return_tensors  # accepted for call-signature compatibility
        self.processor_calls.append(
            {"text": text, "images": images, "kwargs": kwargs})
        return Batch({"input_ids": Ids(ord(c) for c in text),
                      "pixel_values": Ids([0]), "token_type_ids": Ids([0])})

    def _tokenize(self, text, add_special_tokens=True, return_tensors=None):
        del return_tensors  # accepted for call-signature compatibility
        assert add_special_tokens is False, (
            "the template already emitted the special tokens")
        self.tokenizer_calls.append(text)
        return {"input_ids": Ids(ord(c) for c in text)}

    def batch_decode(self, ids, skip_special_tokens=True,
                     clean_up_tokenization_spaces=True):
        del skip_special_tokens  # the fake emits no special tokens
        assert clean_up_tokenization_spaces is False
        return ["".join(chr(value) for value in ids.row)]


class FakeModel:
    """Serves scripted replies, recording the kwargs of every generate."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.device = "cpu"

    def generate(self, past_key_values=None, do_sample=None,
                 max_new_tokens=None, **inputs):
        self.calls.append({"inputs": inputs, "cache": past_key_values,
                           "do_sample": do_sample,
                           "max_new_tokens": max_new_tokens})
        prompt = inputs["input_ids"]
        return Ids(prompt.row + [ord(c) for c in self.replies.pop(0)])


class FakeCacheModule:
    """Stands in for transformers.cache_utils."""

    class DynamicCache:
        pass


def bare_chat(replies, images=(), system=None, tools=None, geometry=True):
    """A Chat with no weights: scripted processor, model, torch and cache."""
    processor = FakeProcessor(geometry=geometry)
    model = FakeModel(replies)
    chat = object.__new__(ai_models.Chat)
    chat._processor, chat._model, chat._torch = processor, model, FakeTorch()
    # Mirrors Chat.__init__: a vision checkpoint's AutoProcessor nests its
    # tokenizer, a text-only checkpoint's AutoTokenizer IS one.
    chat._tokenizer = getattr(processor, "tokenizer", processor)
    chat._images = list(images)
    chat._tools = list(tools) if tools else None
    chat._image_kwargs = ai_models.vision_budget(processor, 576) if images else {}
    chat._messages = []
    if system:
        chat._messages.append(
            {"role": "system", "content": [{"type": "text", "text": system}]})
    chat._cache = None
    chat._ids = None
    return chat, processor, model


@pytest.fixture(autouse=True)
def _no_cached_models():
    """The weights cache is module-global; no test may inherit another's."""
    ai_models.unload()
    yield
    ai_models.unload()


@pytest.fixture
def fake_cache_utils(monkeypatch):
    """Serve transformers.cache_utils from a stand-in; every other name
    still resolves through the real importlib.

    Patched on `ai_models.importlib`, the name the module itself reads, per
    monkeypatch's rule to patch the name used by the system under test.
    """
    real = ai_models.importlib.import_module

    def import_module(name):
        return FakeCacheModule() if name == "transformers.cache_utils" else real(name)

    monkeypatch.setattr(ai_models, "importlib",
                        types.SimpleNamespace(import_module=import_module))


def fake_transformers(from_pretrained, vision=True):
    """A stand-in transformers module answering the auto-class dispatch.

    `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING` is membership-tested against the
    config's TYPE, exactly as transformers' own auto_factory does, so the
    fake mapping is a container of types.
    """
    config = types.SimpleNamespace()
    mapping = [type(config)] if vision else []
    return types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: config),
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING=mapping,
        AutoProcessor=types.SimpleNamespace(
            from_pretrained=lambda *a, **k: "processor"),
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *a, **k: "tokenizer"),
        AutoModelForImageTextToText=types.SimpleNamespace(
            from_pretrained=from_pretrained),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=from_pretrained))


def an_image(size=(64, 64)):
    return Image.new("RGB", size, (30, 30, 30))


# --- vision budget ----------------------------------------------------------


def test_vision_budget_caps_the_area_at_the_declared_token_geometry():
    # Arrange: 16px patches merged 2-to-a-side, i.e. one token per 32x32
    # block of pixels.
    processor = FakeProcessor()

    # Act
    budget = ai_models.vision_budget(processor, 576)

    # Assert: 576 * 32 * 32 is a 768x768-equivalent area, far under the
    # checkpoint's own 16777216 (a 4096x4096 image, 16384 vision tokens).
    assert budget == {"size": {"shortest_edge": 65536, "longest_edge": 589824}}


def test_the_budget_travels_as_size_never_as_max_pixels():
    # Act
    budget = ai_models.vision_budget(FakeProcessor(), 128)

    # Assert: the image processor drops `max_pixels` unless `min_pixels`
    # accompanies it, so a budget sent that way changes nothing at all.
    assert "max_pixels" not in budget
    assert set(budget["size"]) == {"shortest_edge", "longest_edge"}


def test_the_budget_floor_never_exceeds_its_ceiling():
    # Arrange: a budget smaller than the checkpoint's own minimum area.

    # Act
    budget = ai_models.vision_budget(FakeProcessor(), 4)

    # Assert
    size = budget["size"]
    assert size["longest_edge"] == 4 * 1024
    assert size["shortest_edge"] <= size["longest_edge"]


def test_undeclared_geometry_keeps_the_checkpoint_default_and_says_so(caplog):
    # Arrange
    caplog.set_level(logging.WARNING)
    processor = FakeProcessor(geometry=False)

    # Act
    budget = ai_models.vision_budget(processor, 576)

    # Assert: the checkpoint default applies, and the cost is not hidden.
    assert budget == {}
    assert any("vision budget" in message for message in caplog.messages)


# --- cached turn mechanics --------------------------------------------------


def test_the_first_turn_processes_the_image_and_later_turns_do_not(fake_cache_utils):
    # Arrange
    chat, processor, model = bare_chat(["A.", "B.", "C."], images=[an_image()])

    # Act
    chat.ask("one")
    chat.ask("two")
    chat.ask("three")

    # Assert: the image met the processor exactly once, so the vision tower
    # ran once for three questions.
    assert len(processor.processor_calls) == 1
    assert processor.processor_calls[0]["images"] == chat._images
    assert len(processor.tokenizer_calls) == 2
    assert all("pixel_values" not in call["inputs"] for call in model.calls[1:])


def test_the_vision_budget_reaches_the_processor(fake_cache_utils):
    # Arrange
    chat, processor, _ = bare_chat(["A."], images=[an_image()])

    # Act
    chat.ask("one")

    # Assert
    assert processor.processor_calls[0]["kwargs"]["size"] == {
        "shortest_edge": 65536, "longest_edge": 589824}


def test_the_image_placeholders_belong_to_the_first_user_turn_only(fake_cache_utils):
    # Arrange
    chat, processor, _ = bare_chat(["A.", "B."], images=[an_image(), an_image()])

    # Act
    chat.ask("one")
    chat.ask("two")

    # Assert: two placeholders on turn one, none added afterwards.
    assert processor.renders[0]["text"].count("<img>") == 2
    assert processor.renders[-1]["text"].count("<img>") == 2


def test_a_later_turn_carries_the_whole_sequence_not_just_the_delta(fake_cache_utils):
    # Arrange: passing only the new tokens alongside a populated cache
    # slices to zero inside generate, which raises. The full sequence goes.
    chat, processor, model = bare_chat(["Reply one.", "Reply two."],
                                       images=[an_image()])

    # Act
    chat.ask("one")
    first_output = chat._ids
    assert first_output is not None
    chat.ask("two")

    # Assert
    second_input = model.calls[1]["inputs"]["input_ids"]
    delta = processor.tokenizer_calls[-1]
    carried = len(first_output.row)
    assert second_input.row[:carried] == first_output.row
    assert second_input.row[carried:] == [ord(c) for c in delta]


def test_the_delta_is_only_the_text_the_template_added(fake_cache_utils):
    # Arrange
    chat, processor, _ = bare_chat(["Reply one.", "Reply two."],
                                   images=[an_image()])

    # Act
    chat.ask("first question")
    chat.ask("second question")

    # Assert: the delta is the new user turn and the generation prompt, and
    # nothing that the cache already holds.
    assert processor.tokenizer_calls[-1] == "<user:second question><gen>"
    assert "first question" not in processor.tokenizer_calls[-1]


def test_every_turn_shares_one_cache_object(fake_cache_utils):
    # Arrange
    chat, _, model = bare_chat(["A.", "B.", "C."], images=[an_image()])

    # Act
    chat.ask("one")
    chat.ask("two")
    chat.ask("three")

    # Assert
    caches = [call["cache"] for call in model.calls]
    assert isinstance(caches[0], FakeCacheModule.DynamicCache)
    assert all(cache is caches[0] for cache in caches)


def test_decoding_is_greedy_so_a_review_is_reproducible(fake_cache_utils):
    # Arrange
    chat, _, model = bare_chat(["A."], images=[an_image()])

    # Act
    chat.ask("one", max_new_tokens=17)

    # Assert
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 17


def test_only_the_newly_generated_tokens_become_the_reply(fake_cache_utils):
    # Arrange
    chat, _, _ = bare_chat(["the reply"], images=[an_image()])

    # Act
    reply = chat.ask("a question")

    # Assert
    assert reply == "the reply"


def test_the_reply_is_recorded_as_the_assistant_turn(fake_cache_utils):
    # Arrange
    chat, processor, _ = bare_chat(["first reply", "second reply"],
                                   images=[an_image()])

    # Act
    chat.ask("one")
    chat.ask("two")

    # Assert: the reply must re-render, or the cached prefix stops matching.
    assert "<assistant:first reply>" in processor.renders[-1]["text"]


def test_a_processor_key_generate_will_not_accept_is_dropped(fake_cache_utils):
    # Arrange
    chat, _, model = bare_chat(["A."], images=[an_image()])

    # Act
    chat.ask("one")

    # Assert
    assert "token_type_ids" not in model.calls[0]["inputs"]


def test_a_system_turn_is_rendered_first(fake_cache_utils):
    # Arrange: the templates test messages[0].role == 'system' and ignore a
    # system turn found at any other index.
    chat, processor, _ = bare_chat(["A."], images=[an_image()],
                                   system="be terse")

    # Act
    chat.ask("one")

    # Assert
    assert chat._messages[0]["role"] == "system"
    assert processor.renders[0]["text"].startswith("<system:be terse>")


def test_a_chat_without_images_asks_the_processor_for_none(fake_cache_utils):
    # Arrange: text-only work still takes the same first-turn path, because
    # the processor handles text as well as pixels.
    chat, processor, _ = bare_chat(["A."])

    # Act
    chat.ask("text only")

    # Assert: no images, and no vision budget to apply to them.
    assert processor.processor_calls[0]["images"] is None
    assert chat._image_kwargs == {}


# --- the fixed-tools contract -----------------------------------------------


def test_the_same_tools_are_rendered_on_every_turn(fake_cache_utils):
    # Arrange: tools render into the system block at position zero, so
    # changing them mid-conversation rewrites the prefix the cache holds.
    tools = [ai_models.tool("report", "Report.", {"type": "object"})]
    chat, processor, _ = bare_chat(["A.", "B."], images=[an_image()], tools=tools)

    # Act
    chat.ask("one")
    chat.ask("two")

    # Assert
    assert all(render["tools"] == tools for render in processor.renders)


def test_the_cached_render_stays_a_prefix_of_the_full_render(fake_cache_utils):
    # Arrange
    tools = [ai_models.tool("report", "Report.", {"type": "object"})]
    chat, processor, _ = bare_chat(["A.", "B."], images=[an_image()],
                                   system="be terse", tools=tools)

    # Act
    chat.ask("one")
    chat.ask("two")

    # Assert: the delta is taken by string offset, so this is load-bearing.
    assert processor.renders[-1]["text"].startswith(processor.renders[-2]["text"])


def test_ask_json_refuses_when_no_tools_were_declared(fake_cache_utils):
    # Arrange
    chat, _, _ = bare_chat(["A."], images=[an_image()])

    # Act / Assert
    with pytest.raises(ValueError, match="tools declared"):
        chat.ask_json("anything")


def test_ask_json_returns_the_tool_call_arguments(fake_cache_utils):
    # Arrange
    reply = '<tool_call>{"name": "report", "arguments": {"shapes": 2}}</tool_call>'
    chat, _, _ = bare_chat([reply], images=[an_image()],
                           tools=[ai_models.tool("report", "Report.", {"type": "object"})])

    # Act
    payload = chat.ask_json("call report")

    # Assert
    assert payload == {"shapes": 2}


def test_ask_json_retries_an_unusable_reply_then_gives_up(fake_cache_utils):
    # Arrange
    chat, processor, _ = bare_chat(["not json at all", "still not json"],
                                   images=[an_image()],
                                   tools=[ai_models.tool("report", "Report.",
                                                   {"type": "object"})])

    # Act / Assert
    with pytest.raises(ValueError, match="no JSON object"):
        chat.ask_json("call report", attempts=2)
    assert "That was not a tool call." in processor.renders[-1]["text"]


def test_ask_json_accepts_a_reply_on_the_second_attempt(fake_cache_utils):
    # Arrange
    chat, _, _ = bare_chat(["junk", '{"name": "report", "arguments": {"ok": true}}'],
                           images=[an_image()],
                           tools=[ai_models.tool("report", "Report.", {"type": "object"})])

    # Act
    payload = chat.ask_json("call report", attempts=2)

    # Assert
    assert payload == {"ok": True}


def test_tool_builds_the_schema_shape_the_templates_expect():
    # Arrange
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

    # Act
    definition = ai_models.tool("locate", "Give a box.", schema)

    # Assert
    assert definition == {"type": "function", "function": {
        "name": "locate", "description": "Give a box.", "parameters": schema}}


# --- extraction -------------------------------------------------------------


def test_a_brace_inside_a_string_does_not_truncate_the_object():
    # Arrange
    text = 'noise {"caption": "a {curly} brace", "n": 1} trailing'

    # Act / Assert
    assert ai_models.extract_json_object(text) == {"caption": "a {curly} brace", "n": 1}


def test_an_escaped_quote_does_not_end_the_string():
    # Arrange
    text = r'{"caption": "he said \"hi\" {", "n": 2}'

    # Act / Assert
    assert ai_models.extract_json_object(text) == {"caption": 'he said "hi" {', "n": 2}


def test_the_first_balanced_object_wins_over_a_broken_earlier_one():
    # Arrange
    text = '{not json} then {"good": true}'

    # Act / Assert
    assert ai_models.extract_json_object(text) == {"good": True}


def test_text_with_no_object_is_rejected():
    # Act / Assert
    with pytest.raises(ValueError, match="no JSON object"):
        ai_models.extract_json_object("there is no object here")


def test_tool_arguments_unwraps_a_tool_call_block():
    # Arrange
    text = 'sure\n<tool_call>\n{"name": "x", "arguments": {"a": 1}}\n</tool_call>'

    # Act / Assert
    assert ai_models.tool_arguments(text) == {"a": 1}


def test_tool_arguments_accepts_bare_json():
    # Act / Assert
    assert ai_models.tool_arguments('{"a": 1}') == {"a": 1}


def test_tool_arguments_parses_stringified_arguments():
    # Arrange: some checkpoints emit `arguments` as a JSON string.
    text = '{"name": "x", "arguments": "{\\"a\\": 1}"}'

    # Act / Assert
    assert ai_models.tool_arguments(text) == {"a": 1}


def test_tool_arguments_reads_a_tool_call_cut_short_by_the_token_limit():
    # Arrange
    text = '<tool_call>\n{"name": "x", "arguments": {"a": 1}}'

    # Act / Assert
    assert ai_models.tool_arguments(text) == {"a": 1}


# --- resolution -------------------------------------------------------------


def test_a_directory_under_the_models_dir_is_loaded_offline(tmp_path):
    # Arrange
    (tmp_path / "Qwen" / "Qwen3-VL-2B-Instruct").mkdir(parents=True)

    # Act
    location, local_only = ai_models._weights_location(
        "Qwen/Qwen3-VL-2B-Instruct", str(tmp_path))

    # Assert
    assert local_only is True
    assert location == str(tmp_path / "Qwen" / "Qwen3-VL-2B-Instruct")


def test_a_flat_directory_named_for_the_leaf_also_counts(tmp_path):
    # Arrange
    (tmp_path / "Qwen3-VL-2B-Instruct").mkdir()

    # Act
    location, local_only = ai_models._weights_location(
        "Qwen/Qwen3-VL-2B-Instruct", str(tmp_path))

    # Assert
    assert local_only is True
    assert location == str(tmp_path / "Qwen3-VL-2B-Instruct")


def test_an_unprovisioned_reference_stays_a_hub_repo_id(tmp_path):
    # Act
    location, local_only = ai_models._weights_location(
        "Qwen/Qwen3-VL-2B-Instruct", str(tmp_path))

    # Assert
    assert (location, local_only) == ("Qwen/Qwen3-VL-2B-Instruct", False)


def test_an_explicit_device_wins_over_the_environment(monkeypatch):
    # Arrange
    monkeypatch.setenv("AI_DAM_DEVICE", "cuda:1")

    # Act / Assert
    assert ai_models.resolve_device("CPU") == "cpu"


def test_the_environment_device_is_used_when_nothing_is_passed(monkeypatch):
    # Arrange
    monkeypatch.setenv("AI_DAM_DEVICE", "  CUDA:1  ")

    # Act / Assert
    assert ai_models.resolve_device() == "cuda:1"


def test_no_attention_backend_means_transformers_chooses(monkeypatch):
    # Arrange
    monkeypatch.delenv("AI_DAM_ATTN", raising=False)

    # Act / Assert
    assert ai_models.resolve_attn() is None


def test_an_attention_backend_may_name_a_hub_kernel(monkeypatch):
    # Arrange: the pip package ships no Windows wheels, so a Hub kernel is
    # the only route to FlashAttention here.
    monkeypatch.setenv("AI_DAM_ATTN", "kernels-community/flash-attn2")

    # Act / Assert
    assert ai_models.resolve_attn() == "kernels-community/flash-attn2"


def test_weights_are_cached_per_model_device_and_backend(monkeypatch):
    # Arrange
    loaded = []

    def from_pretrained(location, **options):
        loaded.append((location, options.get("attn_implementation")))
        return types.SimpleNamespace(to=lambda device: None, eval=lambda: None)

    def import_module(name):
        if name != "transformers":
            return types.SimpleNamespace()
        return fake_transformers(from_pretrained, vision=True)

    monkeypatch.setattr(ai_models, "importlib",
                        types.SimpleNamespace(import_module=import_module))

    # Act
    ai_models.load("some/model", device="cpu")
    ai_models.load("some/model", device="cpu")
    ai_models.load("some/model", device="cpu", attn="kernels-community/flash-attn2")

    # Assert: the repeat is served from cache; a different backend is a
    # different entry, because it is a different set of weights in memory.
    assert len(loaded) == 2
    assert loaded[0][1] is None
    assert loaded[1][1] == "kernels-community/flash-attn2"


def test_a_missing_runtime_is_reported_as_unavailable(monkeypatch):
    # Arrange
    def import_module(name):
        raise ImportError("no module named torch")

    monkeypatch.setattr(ai_models, "importlib",
                        types.SimpleNamespace(import_module=import_module))

    # Act / Assert: callers report the capability off rather than crashing.
    with pytest.raises(ai_models.ModelUnavailable, match="runtime unavailable"):
        ai_models.load("some/model", device="cpu")


def test_absent_weights_are_reported_as_unavailable(monkeypatch):
    # Arrange
    def absent(*args, **kwargs):
        raise OSError("no such file")

    def import_module(name):
        if name != "transformers":
            return types.SimpleNamespace()
        return types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=absent),
            AutoModelForImageTextToText=types.SimpleNamespace(from_pretrained=absent))

    monkeypatch.setattr(ai_models, "importlib",
                        types.SimpleNamespace(import_module=import_module))

    # Act / Assert
    with pytest.raises(ai_models.ModelUnavailable, match="cannot load"):
        ai_models.load("some/model", device="cpu")


def test_unload_drops_every_cached_model():
    # Arrange
    ai_models._cache[("a", "", "cpu", None)] = ("processor", "model")

    # Act
    ai_models.unload()

    # Assert
    assert ai_models._cache == {}


# --- text-only checkpoints ---------------------------------------------------


class FakeTokenizer(FakeProcessor):
    """A bare AutoTokenizer: no nested `.tokenizer`, no image processor.

    A text-only checkpoint loads one of these instead of an AutoProcessor,
    so Chat must reach the tokenizer through one accessor rather than
    assuming the nested shape.
    """

    def __init__(self):
        super().__init__(geometry=False)
        del self.tokenizer
        del self.image_processor

    def __call__(self, text="", images=None, return_tensors=None, **kwargs):
        assert images is None, "a text-only checkpoint takes no images"
        return super().__call__(text=text, images=None,
                                return_tensors=return_tensors, **kwargs)


def test_a_bare_tokenizer_serves_as_its_own_tokenizer(fake_cache_utils):
    # Arrange: the loader hands back an AutoTokenizer for a text checkpoint.
    tokenizer = FakeTokenizer()
    model = FakeModel(["first answer", "second answer"])
    chat = object.__new__(ai_models.Chat)
    chat._processor, chat._model, chat._torch = tokenizer, model, FakeTorch()
    chat._tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    chat._images, chat._tools, chat._image_kwargs = [], None, {}
    chat._messages, chat._cache, chat._ids = [], None, None

    # Act: two turns, so the second exercises the delta path.
    first = chat.ask("who are you")
    second = chat.ask("and again")

    # Assert
    assert (first, second) == ("first answer", "second answer")
    # A bare tokenizer IS the callable, so both turns go through __call__;
    # the second one carries only the delta.
    assert tokenizer.processor_calls[1]["text"] == "<user:and again><gen>"


def test_a_text_checkpoint_gets_no_vision_budget():
    # Arrange: no declared patch geometry, because there is no vision tower.
    tokenizer = FakeTokenizer()

    # Act / Assert: the budget is a vision concept; a text model has none.
    assert not hasattr(tokenizer, "image_processor")
    assert ai_models.vision_budget(tokenizer, 576) == {}


def test_a_text_only_config_loads_through_the_causal_lm_class(monkeypatch):
    # Arrange: a config absent from the image-text-to-text mapping.
    loaded = []

    def from_pretrained(location, **options):
        loaded.append(location)
        return types.SimpleNamespace(to=lambda device: None, eval=lambda: None)

    fake = fake_transformers(from_pretrained, vision=False)
    monkeypatch.setattr(ai_models, "importlib", types.SimpleNamespace(
        import_module=lambda name: (fake if name == "transformers"
                                    else types.SimpleNamespace())))

    # Act
    processor, _model = ai_models.load("some/text-model", device="cpu")

    # Assert: the tokenizer, not the processor, and the weights still loaded.
    assert processor == "tokenizer"
    assert loaded == ["some/text-model"]
