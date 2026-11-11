"""Observe registry prompts without changing arguments, inputs or embeddings."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps

BGE_QUERY = "Represent this sentence for searching relevant passages: "
QWEN_QUERY = (
    "Instruct: Given a question, retrieve questions that are semantically equivalent "
    "to the given question\nQuery:"
)
QWEN_STS = "Instruct: Retrieve semantically similar text.\nQuery:"
PLAIN = ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2")


def expected(cell, role):
    """Independent smoke oracle from MTEB 2.26.25 sources, not the loaded wrapper."""
    if cell["task"] not in ("STSBenchmark", "NanoQuoraRetrieval"):
        raise ValueError("prompt smoke requires STSBenchmark or NanoQuoraRetrieval")
    if cell["model"] in PLAIN or role == "document":
        return ""
    if cell["model"] == "BAAI/bge-small-en-v1.5":
        return BGE_QUERY if role == "query" else ""
    if cell["model"] == "Qwen/Qwen3-Embedding-0.6B":
        return QWEN_QUERY if role == "query" else QWEN_STS
    raise ValueError("no independent prompt oracle for this model")


@contextmanager
def replace(obj, name, value):
    """Restore the exact instance state, including inherited methods, after any error."""
    present = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if present:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def effective_prompt(model, kwargs):
    prompt = kwargs.get("prompt")
    if prompt is not None:
        return prompt
    name = kwargs.get("prompt_name") or model.default_prompt_name
    return model.prompts[name] if name is not None else None


class Observer:
    def __init__(self, cell, model):
        self.cell, self.model = cell, model
        self.smoke = cell["id"].startswith("smoke-ref-")
        self.role = None
        self.prompt = None
        self.active = False
        self.records = {}

    def wrapper(self, original):
        @wraps(original)
        def observed(*args, **kwargs):
            role = kwargs.get("prompt_type")
            self.role = getattr(role, "value", role)
            return original(*args, **kwargs)

        return observed

    def encoding(self, original, name):
        @wraps(original)
        def observed(*args, **kwargs):
            # encode_query/document resolve their defaults before calling encode.
            if name != "encode":
                return original(*args, **kwargs)
            self.prompt = effective_prompt(self.model, kwargs)
            if self.smoke and (self.prompt or "") != expected(self.cell, self.role):
                raise ValueError(f"unexpected smoke prompt for {self.role}: {self.prompt!r}")
            self.active = True
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False

        return observed

    def preprocessing(self, original):
        @wraps(original)
        def observed(inputs, *args, **kwargs):
            features = original(inputs, *args, **kwargs)
            if self.active and inputs:
                prompt = args[0] if args else kwargs.get("prompt")
                if self.smoke and (prompt or "") != expected(self.cell, self.role):
                    raise ValueError("preprocessing prompt differs from smoke expectation")
                # ST 6.2 InputModule prepends this prefix to each text. Features
                # come from that very call, not from a second tokenizer invocation.
                self.record([(prompt or "") + text for text in inputs], features)
            return features

        return observed

    def record(self, texts, features):
        prompt = self.prompt or ""
        counts = [text.count(prompt) if prompt else 0 for text in texts]
        if (
            self.smoke
            and prompt
            and any(
                count != 1 or not text.startswith(prompt)
                for text, count in zip(texts, counts, strict=True)
            )
        ):
            raise ValueError("smoke prompt prefix must occur exactly once at the start")
        key = (self.role, self.prompt)
        if key not in self.records:
            self.records[key] = {
                "prompt_type": self.role,
                "prompt": self.prompt,
                "first_token_ids": features["input_ids"][0, :8].tolist(),
                "prefix_occurrences": counts[0],
                "prefix_occurrences_min": min(counts),
                "prefix_occurrences_max": max(counts),
            }
        record = self.records[key]
        record["prefix_occurrences_min"] = min(record["prefix_occurrences_min"], min(counts))
        record["prefix_occurrences_max"] = max(record["prefix_occurrences_max"], max(counts))

    def finish(self):
        if self.smoke:
            roles = {role for role, _ in self.records}
            required = {None} if self.cell["task"] == "STSBenchmark" else {"query", "document"}
            if roles != required:
                raise ValueError(
                    f"missing smoke prompt observations: expected {required}, got {roles}"
                )
        return list(self.records.values())


@contextmanager
def capture(cell, wrapper):
    records = []
    if not cell.get("mteb_loader"):
        yield records
        return
    observer = Observer(cell, wrapper.model)
    with ExitStack() as stack:
        stack.enter_context(replace(wrapper, "encode", observer.wrapper(wrapper.encode)))
        for name in ("encode", "encode_query", "encode_document"):
            original = getattr(wrapper.model, name)
            stack.enter_context(replace(wrapper.model, name, observer.encoding(original, name)))
        stack.enter_context(
            replace(wrapper.model, "preprocess", observer.preprocessing(wrapper.model.preprocess))
        )
        yield records
        records.extend(observer.finish())
