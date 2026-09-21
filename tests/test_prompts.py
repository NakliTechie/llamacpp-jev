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
