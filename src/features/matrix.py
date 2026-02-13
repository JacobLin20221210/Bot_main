"""Feature matrix construction."""

from __future__ import annotations

import re
from math import sqrt
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from src.features.embeddings import EMBEDDING_FEATURE_NAMES, build_user_embedding_feature_rows


BASE_FEATURE_NAMES = [
    "tweet_count",
    "z_score",
    "observed_post_count",
    "text_len_mean",
    "text_len_std",
    "text_len_p90",
    "unique_text_ratio",
    "unique_token_ratio",
    "token_entropy",
    "repeated_bigram_ratio",
    "url_rate",
    "mention_rate",
    "hashtag_rate",
    "url_domain_entropy",
    "top_url_domain_share",
    "mention_target_entropy",
    "top_mention_target_share",
    "hashtag_entropy",
    "punctuation_rate",
    "uppercase_rate",
    "digit_rate",
    "retweet_rate",
    "gap_mean_seconds",
    "gap_std_seconds",
    "gap_cv",
    "burst_rate_2m",
    "burst_rate_10m",
    "hourly_entropy",
    "description_len",
    "location_len",
    "username_len",
    "name_len",
    "empty_description",
    "empty_location",
    "gap_entropy",
    "gap_p90_seconds",
    "session_count",
    "session_mean_len",
    "session_max_len",
    "template_repeat_rate",
    "duplicate_window_rate",
    "max_duplicate_streak",
    "template_run_mean_len",
    "periodicity_peak_autocorr",
    "punctuation_rhythm_std",
    "uppercase_rhythm_std",
    "digit_rhythm_std",
    "profile_timeline_token_jaccard",
    "profile_timeline_token_overlap",
    "text_char_entropy_mean",
    "text_char_entropy_std",
    "gap_skewness",
    "gap_kurtosis",
    "day_of_week_entropy",
    "minute_entropy",
    "night_posting_rate",
    "activity_span_seconds",
    "max_idle_gap_seconds",
    "gap_burstiness",
    "cadence_autocorr_24h",
    "yules_k",
    "hapax_ratio",
    "sentence_len_mean",
    "sentence_len_std",
    "function_word_ratio",
    "emoji_rate",
    "repeat_word_ratio",
    "tweet_similarity_mean",
    "sentiment_polarity_std",
    "flesch_reading_ease",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + EMBEDDING_FEATURE_NAMES


TOKEN_PATTERN = re.compile(r"\w+")
SPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)
MENTION_TARGET_PATTERN = re.compile(r"@([A-Za-z0-9_]{1,32})")
HASHTAG_TOKEN_PATTERN = re.compile(r"#(\w+)")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", re.UNICODE)
ALPHA_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]+")
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouyàâäéèêëîïôöùûüÿœ]+", re.IGNORECASE)

FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have", "he", "her",
    "his", "i", "in", "is", "it", "its", "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "with", "you", "your", "yours", "not", "if", "but", "so",
    "de", "des", "du", "la", "le", "les", "un", "une", "et", "ou", "dans", "sur", "avec", "pour", "par",
    "au", "aux", "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "elle", "il", "ils",
    "elles", "nous", "vous", "je", "tu", "ne", "pas", "que", "qui", "quoi", "dont", "où", "est", "sont",
}

POSITIVE_WORDS = {
    "good", "great", "love", "nice", "happy", "best", "excellent", "amazing", "cool", "positive", "success",
    "bon", "bien", "super", "genial", "heureux", "meilleur", "excellent", "positif", "reussi", "succes",
}

NEGATIVE_WORDS = {
    "bad", "hate", "awful", "worst", "sad", "angry", "terrible", "negative", "fail", "failure", "toxic",
    "mauvais", "nul", "triste", "colere", "horrible", "pire", "negatif", "echec", "toxique",
}


def _parse_timestamp(raw: str) -> float | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except ValueError:
        return None


def _safe_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    probs = np.array([count / len(values) for count in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _hourly_entropy(timestamps: list[float]) -> float:
    if not timestamps:
        return 0.0
    hours = [datetime.utcfromtimestamp(ts).hour for ts in timestamps]
    return _safe_entropy([str(hour) for hour in hours])


def _day_of_week_entropy(timestamps: list[float]) -> float:
    if not timestamps:
        return 0.0
    weekdays = [datetime.utcfromtimestamp(ts).weekday() for ts in timestamps]
    return _safe_entropy([str(day) for day in weekdays])


def _minute_entropy(timestamps: list[float]) -> float:
    if not timestamps:
        return 0.0
    minutes = [datetime.utcfromtimestamp(ts).minute for ts in timestamps]
    return _safe_entropy([str(minute) for minute in minutes])


def _night_posting_rate(timestamps: list[float]) -> float:
    if not timestamps:
        return 0.0
    night_posts = sum(datetime.utcfromtimestamp(ts).hour < 6 for ts in timestamps)
    return float(night_posts / max(1, len(timestamps)))


def _safe_skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    std = float(np.std(values))
    if std < 1e-9:
        return 0.0
    centered = (values - float(np.mean(values))) / std
    return float(np.mean(centered**3))


def _safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    std = float(np.std(values))
    if std < 1e-9:
        return 0.0
    centered = (values - float(np.mean(values))) / std
    return float(np.mean(centered**4) - 3.0)


def _gap_burstiness(deltas: np.ndarray) -> float:
    if deltas.size < 2:
        return 0.0
    mean = float(np.mean(deltas))
    std = float(np.std(deltas))
    denom = mean + std
    if denom <= 1e-9:
        return 0.0
    return float((std - mean) / denom)


def _cadence_autocorr_24h(timestamps: list[float]) -> float:
    if len(timestamps) < 4:
        return 0.0
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    span_hours = int((max_ts - min_ts) // 3600) + 1
    lag = 24
    if span_hours <= (lag + 1):
        return 0.0

    hourly = np.zeros(span_hours, dtype=float)
    for ts in timestamps:
        idx = int((ts - min_ts) // 3600)
        if 0 <= idx < span_hours:
            hourly[idx] += 1.0

    left = hourly[:-lag]
    right = hourly[lag:]
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std < 1e-9 or right_std < 1e-9:
        return 0.0

    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _repeated_bigram_ratio(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = [(tokens[idx], tokens[idx + 1]) for idx in range(len(tokens) - 1)]
    return 1.0 - (len(set(bigrams)) / max(1, len(bigrams)))


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    chars = list(text)
    return _safe_entropy(chars)


def _template_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"https?://\S+|www\.\S+", " __url__ ", lowered)
    lowered = re.sub(r"@\w+", " __mention__ ", lowered)
    lowered = re.sub(r"#\w+", " __hashtag__ ", lowered)
    lowered = re.sub(r"\d+", " __number__ ", lowered)
    lowered = SPACE_PATTERN.sub(" ", lowered).strip()
    return lowered


def _yules_k(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    token_counts = Counter(tokens)
    n = float(len(tokens))
    freq_of_freq = Counter(token_counts.values())
    m2 = float(sum((freq**2) * count for freq, count in freq_of_freq.items()))
    return float(1e4 * (m2 - n) / (n * n + 1e-12))


def _sentence_lengths(texts: list[str]) -> np.ndarray:
    lengths: list[float] = []
    for text in texts:
        if not text:
            continue
        parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
        if not parts:
            parts = [text]
        for part in parts:
            token_count = len(TOKEN_PATTERN.findall(part.lower()))
            if token_count > 0:
                lengths.append(float(token_count))
    if not lengths:
        return np.array([0.0], dtype=float)
    return np.asarray(lengths, dtype=float)


def _emoji_rate(texts: list[str]) -> float:
    total_chars = sum(len(text) for text in texts)
    if total_chars <= 0:
        return 0.0
    emoji_count = sum(len(EMOJI_PATTERN.findall(text)) for text in texts)
    return float(emoji_count / total_chars)


def _repeat_word_ratio(texts: list[str]) -> float:
    repeated = 0
    transitions = 0
    for text in texts:
        words = TOKEN_PATTERN.findall(text.lower())
        for left, right in zip(words[:-1], words[1:], strict=False):
            transitions += 1
            if left == right:
                repeated += 1
    if transitions == 0:
        return 0.0
    return float(repeated / transitions)


def _mean_pairwise_token_cosine(texts: list[str]) -> float:
    vectors = [Counter(TOKEN_PATTERN.findall(text.lower())) for text in texts if text]
    if len(vectors) < 2:
        return 0.0

    norms = [sqrt(sum(value * value for value in vector.values())) for vector in vectors]
    similarities: list[float] = []

    for idx in range(len(vectors) - 1):
        if norms[idx] <= 1e-12:
            continue
        for jdx in range(idx + 1, len(vectors)):
            if norms[jdx] <= 1e-12:
                continue
            left = vectors[idx]
            right = vectors[jdx]
            if len(left) > len(right):
                left, right = right, left
            dot = sum(value * right.get(token, 0.0) for token, value in left.items())
            similarities.append(float(dot / (norms[idx] * norms[jdx] + 1e-12)))

    if not similarities:
        return 0.0
    return float(np.mean(similarities))


def _sentiment_polarity_std(texts: list[str]) -> float:
    per_text_polarity: list[float] = []
    for text in texts:
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            continue
        pos = sum(token in POSITIVE_WORDS for token in tokens)
        neg = sum(token in NEGATIVE_WORDS for token in tokens)
        per_text_polarity.append(float((pos - neg) / max(1, len(tokens))))
    if not per_text_polarity:
        return 0.0
    return float(np.std(np.asarray(per_text_polarity, dtype=float)))


def _estimate_syllables(word: str) -> int:
    cleaned = "".join(ALPHA_PATTERN.findall(word.lower()))
    if not cleaned:
        return 0
    groups = VOWEL_GROUP_PATTERN.findall(cleaned)
    syllables = len(groups)
    if cleaned.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def _flesch_reading_ease(texts: list[str]) -> float:
    joined = " ".join(texts).strip()
    if not joined:
        return 0.0

    words = TOKEN_PATTERN.findall(joined.lower())
    if not words:
        return 0.0

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(joined) if part.strip()]
    sentence_count = max(1, len(sentences))
    word_count = len(words)
    syllable_count = sum(_estimate_syllables(word) for word in words)

    score = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllable_count / max(1, word_count)))
    return float(np.clip(score, -100.0, 120.0))


def _duplicate_window_rate(texts: list[str], window: int = 5) -> float:
    if not texts:
        return 0.0
    duplicate_hits = 0
    for idx, text in enumerate(texts):
        if not text:
            continue
        start = max(0, idx - window)
        if text in texts[start:idx]:
            duplicate_hits += 1
    return duplicate_hits / max(1, len(texts))


def _extract_url_domains(texts: list[str]) -> list[str]:
    domains: list[str] = []
    for text in texts:
        for raw_url in URL_PATTERN.findall(text):
            token = raw_url.lower().strip(".,;:!?\")')]")
            if token.startswith("www."):
                token = token[4:]
            elif token.startswith("http://"):
                token = token[7:]
            elif token.startswith("https://"):
                token = token[8:]
            domain = token.split("/", 1)[0]
            if domain:
                domains.append(domain)
    return domains


def _extract_mention_targets(texts: list[str]) -> list[str]:
    targets: list[str] = []
    for text in texts:
        targets.extend(token.lower() for token in MENTION_TARGET_PATTERN.findall(text))
    return targets


def _extract_hashtag_tokens(texts: list[str]) -> list[str]:
    tags: list[str] = []
    for text in texts:
        tags.extend(token.lower() for token in HASHTAG_TOKEN_PATTERN.findall(text))
    return tags


def _top_share(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return float(max(counts.values()) / max(1, len(values)))


def _template_run_lengths(templates: list[str]) -> list[int]:
    if not templates:
        return []
    run_lengths: list[int] = []
    current_template = templates[0]
    current_len = 1
    for template in templates[1:]:
        if template == current_template:
            current_len += 1
        else:
            run_lengths.append(current_len)
            current_template = template
            current_len = 1
    run_lengths.append(current_len)
    return run_lengths


def _periodicity_peak_autocorr(deltas: np.ndarray) -> float:
    if deltas.size < 4:
        return 0.0
    centered = deltas - float(np.mean(deltas))
    std = float(np.std(centered))
    if std < 1e-9:
        return 1.0
    max_lag = min(8, centered.size - 1)
    best = 0.0
    for lag in range(1, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        if left.size < 2 or right.size < 2:
            continue
        denom = float(np.std(left) * np.std(right))
        if denom < 1e-12:
            continue
        corr = float(np.mean((left - np.mean(left)) * (right - np.mean(right))) / denom)
        best = max(best, abs(corr))
    return float(best)


def _session_lengths(timestamps: list[float], session_gap_seconds: float = 30 * 60) -> list[int]:
    if not timestamps:
        return []
    if len(timestamps) == 1:
        return [1]

    lengths: list[int] = []
    current = 1
    for prev_ts, next_ts in zip(timestamps[:-1], timestamps[1:], strict=False):
        if (next_ts - prev_ts) <= session_gap_seconds:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def _profile_timeline_overlap(profile: str, tokens: list[str]) -> tuple[float, float]:
    profile_tokens = set(TOKEN_PATTERN.findall(profile.lower()))
    timeline_tokens = set(tokens)
    if not profile_tokens or not timeline_tokens:
        return 0.0, 0.0
    overlap = profile_tokens.intersection(timeline_tokens)
    union = profile_tokens.union(timeline_tokens)
    jaccard = len(overlap) / max(1, len(union))
    overlap_ratio = len(overlap) / max(1, len(profile_tokens))
    return float(jaccard), float(overlap_ratio)


def extract_user_features(user: dict[str, Any], posts: list[dict[str, Any]]) -> np.ndarray:
    ordered_posts = sorted(posts, key=_post_sort_key)
    texts = [SPACE_PATTERN.sub(" ", str(post.get("text") or "")).strip() for post in ordered_posts]
    text_count = len(texts)
    tokens = [token for text in texts for token in TOKEN_PATTERN.findall(text.lower())]

    text_lengths = np.array([len(text) for text in texts], dtype=float) if texts else np.array([0.0])
    unique_text_ratio = len(set(texts)) / max(1, text_count)
    unique_token_ratio = len(set(tokens)) / max(1, len(tokens))
    token_counts = Counter(tokens)
    hapax_ratio = float(sum(count == 1 for count in token_counts.values()) / max(1, len(tokens)))

    sentence_lengths = _sentence_lengths(texts)
    sentence_len_mean = float(np.mean(sentence_lengths))
    sentence_len_std = float(np.std(sentence_lengths))
    function_word_ratio = float(sum(token in FUNCTION_WORDS for token in tokens) / max(1, len(tokens)))
    emoji_rate = _emoji_rate(texts)
    repeat_word_ratio = _repeat_word_ratio(texts)
    tweet_similarity_mean = _mean_pairwise_token_cosine(texts)
    sentiment_polarity_std = _sentiment_polarity_std(texts)
    flesch_reading_ease = _flesch_reading_ease(texts)
    yules_k = _yules_k(tokens)

    url_rate = sum(("http" in text.lower()) or ("www." in text.lower()) for text in texts) / max(1, text_count)
    mention_rate = sum("@" in text for text in texts) / max(1, text_count)
    hashtag_rate = sum("#" in text for text in texts) / max(1, text_count)
    url_domains = _extract_url_domains(texts)
    mention_targets = _extract_mention_targets(texts)
    hashtag_tokens = _extract_hashtag_tokens(texts)
    url_domain_entropy = float(_safe_entropy(url_domains))
    top_url_domain_share = float(_top_share(url_domains))
    mention_target_entropy = float(_safe_entropy(mention_targets))
    top_mention_target_share = float(_top_share(mention_targets))
    hashtag_entropy = float(_safe_entropy(hashtag_tokens))

    punctuation_series = np.array(
        [(text.count("!") + text.count("?")) / max(1, len(text)) for text in texts],
        dtype=float,
    ) if texts else np.array([0.0], dtype=float)
    uppercase_series = np.array(
        [sum(ch.isupper() for ch in text) / max(1, len(text)) for text in texts],
        dtype=float,
    ) if texts else np.array([0.0], dtype=float)
    digit_series = np.array(
        [sum(ch.isdigit() for ch in text) / max(1, len(text)) for text in texts],
        dtype=float,
    ) if texts else np.array([0.0], dtype=float)

    punctuation_rate = float(np.mean(punctuation_series))
    uppercase_rate = float(np.mean(uppercase_series))
    digit_rate = float(np.mean(digit_series))
    punctuation_rhythm_std = float(np.std(punctuation_series))
    uppercase_rhythm_std = float(np.std(uppercase_series))
    digit_rhythm_std = float(np.std(digit_series))

    retweet_rate = sum(text.lower().startswith("rt ") for text in texts) / max(1, text_count)

    timestamps = [
        ts
        for ts in (_parse_timestamp(str(post.get("created_at") or "")) for post in ordered_posts)
        if ts is not None
    ]

    deltas = np.array([], dtype=float)
    if len(timestamps) >= 2:
        deltas = np.diff(timestamps)
        gap_mean = float(np.mean(deltas))
        gap_std = float(np.std(deltas))
        gap_cv = gap_std / max(1e-6, gap_mean)
        gap_entropy = float(_safe_entropy([f"{int(delta // 60)}" for delta in deltas]))
        gap_p90 = float(np.percentile(deltas, 90))
        burst_2m = float(np.mean(deltas < 120))
        burst_10m = float(np.mean(deltas < 600))
        gap_skewness = float(_safe_skewness(deltas))
        gap_kurtosis = float(_safe_kurtosis(deltas))
        max_idle_gap = float(np.max(deltas))
        gap_burstiness = float(_gap_burstiness(deltas))
    else:
        gap_mean = 0.0
        gap_std = 0.0
        gap_cv = 0.0
        gap_entropy = 0.0
        gap_p90 = 0.0
        burst_2m = 0.0
        burst_10m = 0.0
        gap_skewness = 0.0
        gap_kurtosis = 0.0
        max_idle_gap = 0.0
        gap_burstiness = 0.0

    activity_span_seconds = float((max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0)
    day_of_week_entropy = float(_day_of_week_entropy(timestamps))
    minute_entropy = float(_minute_entropy(timestamps))
    night_posting_rate = float(_night_posting_rate(timestamps))
    cadence_autocorr_24h = float(_cadence_autocorr_24h(timestamps))

    session_lengths = _session_lengths(timestamps)
    session_count = float(len(session_lengths))
    session_mean_len = float(np.mean(session_lengths)) if session_lengths else 0.0
    session_max_len = float(np.max(session_lengths)) if session_lengths else 0.0

    templates = [_template_text(text) for text in texts if text]
    template_repeat_rate = 1.0 - (len(set(templates)) / max(1, len(templates))) if templates else 0.0
    duplicate_window_rate = _duplicate_window_rate(texts, window=5)
    template_run_lengths = _template_run_lengths(templates)
    max_duplicate_streak = float(np.max(template_run_lengths)) if template_run_lengths else 0.0
    template_run_mean_len = float(np.mean(template_run_lengths)) if template_run_lengths else 0.0
    periodicity_peak_autocorr = float(_periodicity_peak_autocorr(deltas))

    char_entropies = np.array([_char_entropy(text) for text in texts], dtype=float) if texts else np.array([0.0])
    text_char_entropy_mean = float(np.mean(char_entropies))
    text_char_entropy_std = float(np.std(char_entropies))

    description = str(user.get("description") or "").strip()
    location = str(user.get("location") or "").strip()
    username = str(user.get("username") or "")
    name = str(user.get("name") or "")
    profile = f"{name} {description} {location}".strip()
    profile_timeline_jaccard, profile_timeline_overlap = _profile_timeline_overlap(profile, tokens)

    return np.array(
        [
            float(user.get("tweet_count", text_count) or text_count),
            float(user.get("z_score", 0.0) or 0.0),
            float(text_count),
            float(np.mean(text_lengths)),
            float(np.std(text_lengths)),
            float(np.percentile(text_lengths, 90)),
            float(unique_text_ratio),
            float(unique_token_ratio),
            float(_safe_entropy(tokens)),
            float(_repeated_bigram_ratio(tokens)),
            float(url_rate),
            float(mention_rate),
            float(hashtag_rate),
            float(url_domain_entropy),
            float(top_url_domain_share),
            float(mention_target_entropy),
            float(top_mention_target_share),
            float(hashtag_entropy),
            float(punctuation_rate),
            float(uppercase_rate),
            float(digit_rate),
            float(retweet_rate),
            float(gap_mean),
            float(gap_std),
            float(gap_cv),
            float(burst_2m),
            float(burst_10m),
            float(_hourly_entropy(timestamps)),
            float(len(description)),
            float(len(location)),
            float(len(username)),
            float(len(name)),
            float(1 if not description else 0),
            float(1 if not location else 0),
            float(gap_entropy),
            float(gap_p90),
            float(session_count),
            float(session_mean_len),
            float(session_max_len),
            float(template_repeat_rate),
            float(duplicate_window_rate),
            float(max_duplicate_streak),
            float(template_run_mean_len),
            float(periodicity_peak_autocorr),
            float(punctuation_rhythm_std),
            float(uppercase_rhythm_std),
            float(digit_rhythm_std),
            float(profile_timeline_jaccard),
            float(profile_timeline_overlap),
            float(text_char_entropy_mean),
            float(text_char_entropy_std),
            float(gap_skewness),
            float(gap_kurtosis),
            float(day_of_week_entropy),
            float(minute_entropy),
            float(night_posting_rate),
            float(activity_span_seconds),
            float(max_idle_gap),
            float(gap_burstiness),
            float(cadence_autocorr_24h),
            float(yules_k),
            float(hapax_ratio),
            float(sentence_len_mean),
            float(sentence_len_std),
            float(function_word_ratio),
            float(emoji_rate),
            float(repeat_word_ratio),
            float(tweet_similarity_mean),
            float(sentiment_polarity_std),
            float(flesch_reading_ease),
        ],
        dtype=float,
    )


def build_feature_matrix(
    users: list[dict[str, Any]],
    posts_by_author: dict[str, list[dict[str, Any]]],
    embedding_cache_dir: str = "output/cache/embeddings",
    embedding_model_name: str = "intfloat/multilingual-e5-small",
) -> tuple[list[str], np.ndarray]:
    user_ids: list[str] = []
    rows: list[np.ndarray] = []
    for user in users:
        user_id = str(user.get("id", ""))
        user_ids.append(user_id)
        rows.append(extract_user_features(user, posts_by_author.get(user_id, [])))

    if not rows:
        return user_ids, np.empty((0, len(FEATURE_NAMES)), dtype=float)

    base_matrix = np.vstack(rows)
    embedding_matrix = build_user_embedding_feature_rows(
        users=users,
        posts_by_author=posts_by_author,
        cache_dir=embedding_cache_dir,
        model_name=embedding_model_name,
    )
    if embedding_matrix.shape[0] != base_matrix.shape[0]:
        raise RuntimeError("Embedding feature rows mismatch base feature rows")

    return user_ids, np.hstack([base_matrix, embedding_matrix])


def _post_sort_key(post: dict[str, Any]) -> tuple[float, str]:
    raw_ts = str(post.get("created_at") or "")
    ts = _parse_timestamp(raw_ts)
    if ts is None:
        return float("inf"), raw_ts
    return ts, raw_ts


def build_sequence_documents(
    users: list[dict[str, Any]],
    posts_by_author: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    user_ids: list[str] = []
    documents: list[str] = []

    for user in users:
        user_id = str(user.get("id", ""))
        user_posts = sorted(posts_by_author.get(user_id, []), key=_post_sort_key)

        snippets: list[str] = []
        for post in user_posts:
            text = SPACE_PATTERN.sub(" ", str(post.get("text") or "")).strip()
            if not text:
                continue
            lowered = text.lower()
            tags: list[str] = []
            if "http" in lowered or "www." in lowered:
                tags.append("__url__")
            if "@" in text:
                tags.append("__mention__")
            if "#" in text:
                tags.append("__hashtag__")
            snippets.append(f"{text} {' '.join(tags)}".strip())

        description = SPACE_PATTERN.sub(" ", str(user.get("description") or "")).strip()
        location = SPACE_PATTERN.sub(" ", str(user.get("location") or "")).strip()
        profile = f"__desc__ {description} __loc__ {location}".strip()
        timeline = " __post_sep__ ".join(snippets)

        document = f"{profile} __timeline__ {timeline}".strip()
        user_ids.append(user_id)
        documents.append(document)

    return user_ids, documents
