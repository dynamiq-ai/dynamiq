import threading

import pytest
from jinja2 import Environment, TemplateSyntaxError, meta

from dynamiq.prompts import Message, Prompt
from dynamiq.prompts.prompts import get_parameters_for_template
from dynamiq.utils import jinja
from dynamiq.utils.jinja import compile_template, template_parameters


@pytest.fixture(autouse=True)
def empty_caches():
    compile_template.cache_clear()
    template_parameters.cache_clear()
    yield
    compile_template.cache_clear()
    template_parameters.cache_clear()


@pytest.fixture
def compiled_sources(monkeypatch):
    """Records each source jinja actually compiles."""
    sources = []
    original = jinja.Template

    def counting(source):
        sources.append(source)
        return original(source)

    monkeypatch.setattr(jinja, "Template", counting)
    return sources


def test_compile_template_compiles_each_source_once(compiled_sources):
    prompt = Prompt(messages=[Message(content="Answer: {{input}}")])

    for i in range(20):
        assert prompt.format_messages(input=str(i)) == [{"content": f"Answer: {i}", "role": "user"}]

    assert compiled_sources == ["Answer: {{input}}"]


def test_compile_template_returns_the_same_template_for_the_same_source():
    assert compile_template("{{ a }}") is compile_template("{{ a }}")
    assert compile_template("{{ a }}") is not compile_template("{{ b }}")


def test_compile_template_renders_like_jinja():
    source = "{% for item in items %}{{ item | upper }}{% if not loop.last %}, {% endif %}{% endfor %}"

    assert compile_template(source).render(items=["a", "b"]) == "A, B"
    assert compile_template(source).render(items=["c"]) == "C"


def test_compile_template_does_not_cache_failures(compiled_sources):
    for _ in range(3):
        with pytest.raises(TemplateSyntaxError):
            compile_template("{% if %}")

    assert compiled_sources.count("{% if %}") == 3
    assert compile_template.cache_info().currsize == 0


def test_compile_template_is_shared_safely_across_threads():
    errors = []

    def render(i):
        try:
            assert compile_template("{{ value }}-{{ value * 2 }}").render(value=i) == f"{i}-{i * 2}"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=render, args=(i,)) for i in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert compile_template.cache_info().currsize == 1


def test_template_parameters_match_jinja():
    source = "{{ a }} {% for x in items %}{{ x }}{{ b }}{% endfor %}{% set c = 1 %}{{ c }}"
    expected = meta.find_undeclared_variables(Environment(autoescape=True).parse(source))

    assert template_parameters(source) == expected == {"a", "b", "items"}


def test_get_parameters_for_template_returns_a_set_of_its_own():
    parameters = get_parameters_for_template("{{ a }}")
    parameters.add("mutated")

    assert get_parameters_for_template("{{ a }}") == {"a"}
    assert get_parameters_for_template("{{ a }}", env=Environment()) == {"a"}


def test_prompt_required_parameters_are_unchanged():
    prompt = Prompt(
        messages=[
            Message(role="system", content="You are {{ role }}."),
            Message(role="user", content="{{ input }}"),
            Message(role="user", content="{{ ignored }}", static=True),
        ]
    )

    assert prompt.get_required_parameters() == {"role", "input"}
