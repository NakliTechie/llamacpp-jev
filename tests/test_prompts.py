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
    chat = [{"role": "user", "content": [{"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}]
    with pytest.raises(ContentError):
        state_messages(chat, allow_images=False)
    messages, images = state_messages(chat, allow_images=True)
    assert images == ["QUJD"]
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

def _png(width: int, height: int) -> str:
    import base64
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    return base64.b64encode(header).decode()


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
    chat = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]}]
    with pytest.raises(ContentError, match="more than 1 image"):
        state_messages(chat, allow_images=True, max_images=1)


def test_oversized_image_bytes_rejected():
    big = "A" * 200  # est ~150 decoded bytes
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big}"}}]}]
    with pytest.raises(ContentError, match="decoded bytes"):
        state_messages(chat, allow_images=True, max_image_bytes=10)


def test_pixel_bomb_rejected():
    chat = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png(5000, 5000)}"}}]}]
    with pytest.raises(ContentError, match="pixel limit"):
        state_messages(chat, allow_images=True, max_image_pixels=1000)
