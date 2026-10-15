"""Lazy, reusable local inference with the GLiNER 2.0 checkpoint."""
from dataclasses import asdict, dataclass, replace
import math
import re
from collections.abc import Mapping

from .topics import DEFAULT_TOPICS

MODEL_ID = "fastino/gliner2-base-v1"
MAX_CHARACTERS = 1000
TOPIC_POLICY_VERSION = "compact-topics-gun-evidence-v2"
GUN_EVIDENCE = re.compile(r"\b(guns?|gunfire|gunshots?|gunman|gunmen|firearms?|shoot(?:ing|ings|er|ers|out)?|shot|shots|rifles?|pistols?|handguns?|weapons?|armed|ammunition|ammo|bullets?|NRA|second amendment|2nd amendment)\b", re.I)


@dataclass(frozen=True)
class TopicResult:
    topic: str
    confidence: float
    model: str = MODEL_ID
    reranked: bool = False

    def to_dict(self):
        return asdict(self)


class TopicClassifier:
    """Pick one general topic from a configurable label-to-description mapping."""

    def __init__(self, topics=None, device="cpu"):
        topics = DEFAULT_TOPICS if topics is None else topics
        if not isinstance(topics, Mapping) or len(topics) < 2:
            raise ValueError("Provide at least two topics as a label-to-description mapping.")
        if any(not isinstance(k, str) or not k.strip() or
               not isinstance(v, str) or not v.strip() for k, v in topics.items()):
            raise ValueError("Topic labels and descriptions must be non-empty strings.")
        self.device = device
        self._default_policy = dict(topics) == DEFAULT_TOPICS
        self._topics = dict(topics)
        self._model = None
        self._schema = None

    def _load(self):
        if self._model is None:
            try:
                from gliner2 import AutoExtractor
            except ImportError as exc:
                raise RuntimeError(
                    "Install dependencies: python -m pip install -r requirements-topics.txt"
                ) from exc
            model = AutoExtractor.from_pretrained(MODEL_ID)
            model.to(self.device)
            model.eval()
            schema = model.create_schema().classification("topic", self._topics)
            self._model, self._schema = model, schema
        return self._model

    def classify(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        if len(text) > MAX_CHARACTERS:
            raise ValueError(f"Use a short passage of at most {MAX_CHARACTERS} characters.")
        output = self._load().extract(text, self._schema, include_confidence=True)
        return self._resolve_gun_evidence([text], [self._result(output)], 1)[0]

    def classify_many(self, texts, *, batch_size=8):
        """Classify a batch in input order using the configured taxonomy."""
        texts = list(texts)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if any(not isinstance(text, str) or not text.strip() or
               len(text) > MAX_CHARACTERS for text in texts):
            raise ValueError("Every text must contain 1 to 1,000 characters and not be blank.")
        if not texts:
            return []
        outputs = self._load().batch_extract(texts, self._schema, batch_size=batch_size,
                                             include_confidence=True)
        if len(outputs) != len(texts):
            raise RuntimeError("Model returned the wrong number of results.")
        return self._resolve_gun_evidence(texts, [self._result(output) for output in outputs], batch_size)

    def _resolve_gun_evidence(self, texts, results, batch_size):
        """Rerank unsupported gun predictions; never invent a replacement label.

        This conservative lexical guard only applies to the built-in taxonomy.
        Implicit references may be missed; confidence is conditional on candidates.
        """
        indices = [i for i, (text, result) in enumerate(zip(texts, results))
                   if self._default_policy and result.topic == "gun issues"
                   and not GUN_EVIDENCE.search(text)]
        if indices:
            schema = self._load().create_schema().classification(
                "topic", {k: v for k, v in self._topics.items() if k != "gun issues"})
            outputs = self._load().batch_extract(
                [texts[i] for i in indices], schema, batch_size=batch_size,
                include_confidence=True)
            if len(outputs) != len(indices):
                raise RuntimeError("Model returned the wrong number of reranked results.")
            for i, output in zip(indices, outputs):
                result = self._result(output)
                if result.topic == "gun issues":
                    raise RuntimeError("Model returned an excluded topic.")
                results[i] = replace(result, reranked=True)
        return results

    def _result(self, output):
        try:
            label = output["topic"]["label"]
            confidence = float(output["topic"]["confidence"])
            if label not in self._topics or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError()
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Unexpected GLiNER2 classification response.") from exc
        return TopicResult(label, confidence)
