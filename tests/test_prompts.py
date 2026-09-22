from itertools import pairwise

import pytest
from conftest import ENDING, fake_labels

from llamajev.models import SystemOneRequest
from llamajev.prompts import ContentError, PromptCompiler, build_messages, state_messages


def make_request(**questions):
    return SystemOneRequest(model="jev-latest", state="hello", questions=questions)


def test_plain_state_becomes_one_user_message():
    messages, images = state_messages("hello", allow_images=False)
    assert messages == [{"role": "user", "content": "hello"}]
    assert images == []


def test_json_state_is_serialized_intact():
    messages, _ = state_messages({"ticket": 1, "text": "x"}, allow_images=False)
    assert messages == [{"role": "user", "content": '{"ticket":1,"text":"x"}'}]


def test_messages_envelope_is_unwrapped_only_when_exact():
    chat = [{"role": "user", "content": "hi"}]
    assert state_messages({"messages": chat}, False)[0] == chat
    wrapped, _ = state_messages({"messages": chat, "meta": 1}, False)
    assert wrapped[0]["content"].startswith("{")


def test_bad_role_rejected():
    with pytest.raises(ContentError):
        state_messages([{"role": "robot", "content": "x"}], False)


def test_image_parts_need_vision():
    img = _img()
    chat = [{"role": "user", "content": [{"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]
    with pytest.raises(ContentError):
        state_messages(chat, allow_images=False)
    messages, images = state_messages(chat, allow_images=True)
    assert images == [img]
    assert messages == chat


def test_remote_image_url_rejected():
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
    with pytest.raises(ContentError):
        state_messages(chat, allow_images=True)


def test_compile_splits_prefix_and_renders_options():
    request = make_request(
        team={"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": None}},
        yes={"type": "noul", "instructions": "Refund?"},
        level={"type": "score", "instructions": "Urgency", "criteria": ["Low", "High"]},
    )
    messages, images, marker = build_messages(request, allow_images=False)
    assert messages[-1]["content"].endswith(marker)
    rendered = f"<|im_start|>user\nhello<|im_end|>\n<|im_start|>user\nPREAMBLE\n\n{marker}{ENDING}"
    prepared = PromptCompiler(fake_labels()).compile(request, rendered, marker, images)
    assert prepared.prefix == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>user\nPREAMBLE\n\n"
    team = prepared.branches[0]
    assert team.suffix == "Question: Which team?\n\nOptions:\nA: Payments\nB: tech" + ENDING + "Answer:\n"
    assert team.labels == ["A", "B"] and team.option_keys == ["billing", "tech"]
    assert team.grammar == 'root ::= "A" | "B"'
    assert prepared.branches[1].option_keys == ["true", "false"]
    assert "A: Yes\nB: No" in prepared.branches[1].suffix
    assert prepared.branches[2].option_keys == ["0", "1"]


def test_multiline_description_is_indented():
    request = make_request(q={"type": "choice", "instructions": "x", "criteria": {"a": "line1\nline2", "b": "y"}})
    _, _, marker = build_messages(request, False)
    prepared = PromptCompiler(fake_labels()).compile(request, f"pre {marker}{ENDING}", marker, [])
    assert "A: line1\n   line2\nB: y" in prepared.branches[0].suffix


def test_compiler_requires_64_labels():
    with pytest.raises(ValueError):
        PromptCompiler(fake_labels()[:10])


def test_marker_must_survive_template():
    request = make_request(q={"type": "noul", "instructions": "x"})
    with pytest.raises(ContentError):
        PromptCompiler(fake_labels()).compile(request, "no marker here", "MARK", [])


# --- hardening: message alternation + image bounds ---

def _img(width: int = 4, height: int = 4) -> str:
    """Base64 of a real PNG of the given size (validation now decodes the header via Pillow)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_no_two_consecutive_user_messages_when_state_ends_in_user():
    # State ending in a user turn: the preamble merges into it, not a second user message.
    req = SystemOneRequest(model="jev-latest", state=[{"role": "user", "content": "hi"}],
                           questions={"q": {"type": "noul", "instructions": "x"}})
    messages, _, marker = build_messages(req, allow_images=False)
    roles = [m["role"] for m in messages]
    assert not any(a == b == "user" for a, b in pairwise(roles)), roles
    assert marker in messages[-1]["content"] and "hi" in messages[-1]["content"]


def test_assistant_tail_gets_a_new_user_message():
    req = SystemOneRequest(model="jev-latest", state=[{"role": "user", "content": "hi"},
                                                      {"role": "assistant", "content": "yo"}],
                           questions={"q": {"type": "noul", "instructions": "x"}})
    messages, _, _ = build_messages(req, allow_images=False)
    assert messages[-1]["role"] == "user" and messages[-2]["role"] == "assistant"


def test_too_many_images_rejected():
    img = _img()
    chat = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
    ]}]
    with pytest.raises(ContentError, match="more than 1 image"):
        state_messages(chat, allow_images=True, max_images=1)


def test_non_image_payload_rejected():
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}]
    with pytest.raises(ContentError, match="PNG or JPEG"):
        state_messages(chat, allow_images=True)


def test_lying_mime_still_validated_by_bytes():
    # A real PNG declared as jpeg is fine (bytes win); text declared as an image is rejected.
    ok = _img()
    state_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ok}"}}]}], allow_images=True)
    import base64
    junk = base64.b64encode(b"not an image at all").decode()
    with pytest.raises(ContentError, match="PNG or JPEG"):
        state_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{junk}"}}]}], allow_images=True)


def test_uppercase_mime_and_missing_base64_flag():
    ok = _img()
    # Uppercase media type is valid (RFC 2045 case-insensitive).
    state_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:IMAGE/PNG;base64,{ok}"}}]}], allow_images=True)
    # A ";base64=garbage" flag is not the base64 token → rejected.
    with pytest.raises(ContentError, match="data:image"):
        state_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64=x,{ok}"}}]}], allow_images=True)


def test_ico_rejected_before_decode():
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="ICO")
    ico = base64.b64encode(buf.getvalue()).decode()
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/x-icon;base64,{ico}"}}]}]
    with pytest.raises(ContentError, match="PNG or JPEG"):
        state_messages(chat, allow_images=True)


def test_oversized_image_bytes_rejected():
    big = "A" * 200  # est ~150 decoded bytes
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big}"}}]}]
    with pytest.raises(ContentError, match="decoded bytes"):
        state_messages(chat, allow_images=True, max_image_bytes=10)


def test_pixel_bomb_rejected():
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img(50, 50)}"}}]}]
    with pytest.raises(ContentError, match="pixel limit"):
        state_messages(chat, allow_images=True, max_image_pixels=100)


def test_aggregate_pixel_budget():
    img = _img(50, 50)  # 2500 px each
    chat = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
    ]}]
    with pytest.raises(ContentError, match="request limit"):
        state_messages(chat, allow_images=True, max_image_pixels=3000, max_total_image_pixels=4000)


def test_build_messages_does_not_mutate_state():
    state = [{"role": "user", "content": "hi"}]
    req = SystemOneRequest(model="jev-latest", state=state, questions={"q": {"type": "noul", "instructions": "x"}})
    build_messages(req, allow_images=False)
    build_messages(req, allow_images=False)
    assert req.state == [{"role": "user", "content": "hi"}]  # unchanged, no accumulated preamble
