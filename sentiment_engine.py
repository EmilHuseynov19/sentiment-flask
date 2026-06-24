#!/usr/bin/env python3
"""
sentiment_engine.py — Turkish sentiment analysis engine backed by DuckDB.
Used by the Flask app and the CLI watcher.
"""
from __future__ import annotations

import re
import time
import duckdb
import threading
from pathlib import Path
from functools import lru_cache
from typing import Any

DB_PATH = Path("/Users/macbook/Documents/codex/trendyol_research_mvp/data/trendyol.duckdb")

# ── Turkish sentiment lexicon ───────────────────────────────────────────────
LEXICON: dict[str, float] = {
    # strong positive
    "mükemmel": 2, "mukemmel": 2, "muhteşem": 2, "muhtesem": 2,
    "bayıldım": 2, "bayildim": 2, "bayılıyorum": 2, "bayiliyorum": 2,
    "vazgeçilmez": 2, "vazgecilmez": 2, "şaheser": 2, "saheser": 2,
    "efsane": 2, "inanılmaz": 2, "inanilmaz": 2, "harika": 2,
    "müthiş": 2, "muthis": 2,
    # positive
    "güzel": 1, "guzel": 1, "beğendim": 1, "begendim": 1,
    "memnun": 1, "sevdim": 1, "tavsiye": 1, "kaliteli": 1,
    "başarılı": 1, "basarili": 1, "süper": 1, "super": 1,
    "favori": 1, "kullanıyorum": 1, "kullaniyorum": 1,
    "aldım": 1, "aldim": 1, "memnunum": 1, "teşekkür": 1, "tesekkur": 1,
    "hızlı": 1, "hizli": 1, "sorunsuz": 1, "kalıcı": 1, "kalici": 1,
    "kesinlikle": 0.5, "cidden": 0.5, "gerçekten": 0.5, "gercekten": 0.5,
    # strong negative
    "berbat": -2, "iğrenç": -2, "igrenc": -2, "rezalet": -2,
    "defolu": -2, "kırık": -2, "kirik": -2, "bozuk": -2,
    "yırtık": -2, "yirtik": -2, "şikayet": -2, "sikayet": -2,
    # negative
    "kötü": -1, "kotu": -1, "beğenmedim": -1, "begenmedim": -1,
    "sevmedim": -1, "hasarlı": -1, "hasarli": -1,
    "sorun": -1, "sorunlu": -1, "problem": -1,
    "eksik": -1, "yanlış": -1, "yanlis": -1,
    "geç": -1, "gec": -1, "yavaş": -1, "yavas": -1, "kalitesiz": -1,
    "iade": -1,
}

NEGATORS = {"değil", "degil", "yok", "hiç", "hic", "ne", "ama",
            "fakat", "ancak", "lakin", "rağmen", "ragmen"}
INTENSIFIERS = {"çok", "cok", "aşırı", "asiri", "fazla", "oldukça", "oldukca"}


# ── Thread-safe cache ──────────────────────────────────────────────────────
_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_cache_ts: float = 0
_CACHE_TTL = 30  # seconds


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r'[^a-z0-9çğıöşü\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def analyze_sentiment(text: str) -> str:
    """Score-based Turkish sentiment: 'positive' | 'neutral' | 'negative'."""
    if not text:
        return "neutral"
    t = _normalize(text)
    words = t.split()
    score = 0.0
    negate = False
    found = False
    for i, w in enumerate(words):
        if w in NEGATORS:
            negate = True
            continue
        if w not in LEXICON:
            continue
        val = LEXICON[w]
        if val == 0:
            continue
        found = True
        amp = 1.5 if (i > 0 and words[i - 1] in INTENSIFIERS) else 1.0
        if negate:
            score -= val * amp
            negate = False
        else:
            score += val * amp
    if not found:
        return "neutral"
    return "positive" if score >= 0.5 else "negative" if score <= -0.5 else "neutral"


def _extract_issues(all_texts: list[str]) -> list[str]:
    """Extract common complaint themes."""
    issues: list[tuple[str, int]] = []

    def count(patterns, filter_extra=None) -> int:
        c = 0
        for r in all_texts:
            if not r:
                continue
            if filter_extra and not any(re.search(fe, r, re.IGNORECASE) for fe in filter_extra):
                continue
            if any(re.search(p, r, re.IGNORECASE) for p in patterns):
                c += 1
        return c

    rules = [
        (["ge[çc]", "yavaş", "yavas"], ["karg", "teslimat", "paket", "ambalaj", "kurye"],
         "Çatdırılma gecikməsi / Delivery delay"),
        (["hasar", "kırık", "kirik", "yırt", "yirt", "ezil", "defolu", "lek[ei]", "çatlak", "catlak"],
         ["paket", "kutu", "ambalaj", "ürün", "urun", "gel"],
         "Zədələnmiş qablaşdırma / Damaged packaging"),
        (["defolu", "bozuk", "kırık", "kirik", "hasarlı", "hasarli", "yırtık", "yirtik"],
         [],
         "Məhsul qüsuru / Product defect"),
        (["dar.*gel", "bol.*gel", "beden.*büyük", "beden.*buyuk", "beden.*küçük", "beden.*kucuk",
          "kalıp", "kalip.*dar", "numara.*büyük", "numara.*buyuk"],
         [],
         "Ölçü uyğunsuzluğu / Size mismatch"),
        (["ağır.*koku", "agir.*koku", "rahatsız.*koku", "rahatsiz.*koku", "itici.*koku",
          "baş ağrısı", "bas agrisi"],
         [],
         "Həddindən artıq qoxu / Overpowering scent"),
        (["kalitesiz", "düşük kalite", "dusuk kalite", "ince.*kumaş", "ince.*kumas",
          "kötü.*kumaş", "kotu.*kumas", "basit"],
         [],
         "Keyfiyyət problemi / Quality problem"),
        (["eksik.*gel", "yanlış.*ürün", "yanlis.*urun", "farklı.*ürün", "farkli.*urun",
          "başka.*ürün", "baska.*urun"],
         [],
         "Sifarişdə səhv / Wrong/missing item"),
        (["iade"],
         [],
         "Geri qaytarma / Return problem"),
        (["təlimat", "manual", "kitabç", "kitapc", "kullanma kılavuzu",
          "kullanma kilavuzu"],
         [],
         "İstifadə təlimatı yox / Missing manual"),
    ]

    for patterns, filter_extra, label in rules:
        c = count(patterns, filter_extra)
        if c > 0:
            issues.append((label, c))

    issues.sort(key=lambda x: -x[1])
    return [label for label, _ in issues[:5]]


def _generate_summary(pct_pos: float, pct_neg: float, issues: list[str],
                      product_name: str) -> dict[str, str]:
    brand = product_name.split()[0] if product_name else "Bu məhsul"
    if pct_pos > 70:
        en = f"Customers are highly satisfied with {brand}. "
        az = f"Müştərilər {brand} məhsulundan çox razıdır. "
    elif pct_pos > 50:
        en = f"Most customers are satisfied with {brand}. "
        az = f"Müştərilərin əksəriyyəti {brand} məhsulundan razıdır. "
    else:
        en = f"Opinions on {brand} are mixed. "
        az = f"{brand} haqqında rəylər qarışıqdır. "
    if issues:
        iss = issues[0].split("/")[0].strip()
        en += f"Some users reported issues with {iss.lower()}."
        az += f"Bəzi istifadəçilər {iss.lower()} qeyd edir."
    return {"en": en.strip(), "az": az.strip()}


# ── Public API ──────────────────────────────────────────────────────────────

def get_products() -> list[dict]:
    """Return a lightweight product list for the dropdown."""
    with _cache_lock:
        now = time.time()
        if "product_list" in _cache and (now - _cache.get("_ts", 0)) < _CACHE_TTL:
            return _cache["product_list"]

    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("""
        SELECT DISTINCT p.url, p.name, p.brand, p.rating, p.review_count, p.image_url
        FROM product_observations p
        WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.product_url = p.url)
        ORDER BY p.review_count DESC
    """).fetchall()
    con.close()

    result = [
        {"url": r[0], "name": r[1], "brand": r[2] or "",
         "rating": float(r[3]) if r[3] else 0,
         "reviewCount": int(r[4]) if r[4] else 0,
         "imageUrl": r[5] or ""}
        for r in rows
    ]

    with _cache_lock:
        _cache["product_list"] = result
        _cache["_ts"] = time.time()
    return result


def get_products_with_sentiment(limit: int = 20) -> list[dict]:
    """Return top N products with basic sentiment breakdown pre-computed."""
    products = get_products()[:limit]
    total_needed = len(products)
    if total_needed == 0:
        return []

    # Batch-load all review texts for these products
    urls = [p["url"] for p in products]
    con = duckdb.connect(str(DB_PATH), read_only=True)
    placeholders = ",".join("?" for _ in urls)
    rows = con.execute(
        f"SELECT product_url, review_text FROM reviews "
        f"WHERE product_url IN ({placeholders}) "
        f"AND review_text IS NOT NULL AND review_text != ''",
        urls
    ).fetchall()
    con.close()

    # Group reviews by product_url
    reviews_by_url: dict[str, list[str]] = {u: [] for u in urls}
    for url, text in rows:
        if url in reviews_by_url:
            reviews_by_url[url].append(text)

    # Compute sentiment per product (lightweight — only sentiment, no issues/summary)
    for prod in products:
        texts = reviews_by_url.get(prod["url"], [])
        total = len(texts)
        if total == 0:
            prod["sentiment"] = {"positive": 0, "neutral": 0, "negative": 0}
            prod["totalReviews"] = 0
            continue
        sentiments = [analyze_sentiment(t) for t in texts]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        pct_pos = round(pos / total * 100)
        pct_neg = round(neg / total * 100)
        prod["sentiment"] = {
            "positive": pct_pos,
            "neutral": 100 - pct_pos - pct_neg,
            "negative": pct_neg,
        }
        prod["totalReviews"] = total

    return products


def get_product_sentiment(product_url: str) -> dict | None:
    """Full sentiment analysis for a single product."""
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Product info
    prod = con.execute(
        "SELECT name, brand, price, rating, review_count, image_url "
        "FROM product_observations WHERE url = ? LIMIT 1",
        [product_url]
    ).fetchone()
    if not prod:
        con.close()
        return None

    # Reviews
    texts = [
        r[0] for r in con.execute(
            "SELECT review_text FROM reviews WHERE product_url = ? AND review_text IS NOT NULL AND review_text != ''",
            [product_url]
        ).fetchall()
    ]
    con.close()

    total = len(texts)
    if total == 0:
        return None

    # Sentiment
    sentiments = [analyze_sentiment(t) for t in texts]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = total - pos - neg

    pct_pos = round(pos / total * 100)
    pct_neg = round(neg / total * 100)
    pct_neu = 100 - pct_pos - pct_neg

    issues = _extract_issues(texts)
    summary = _generate_summary(pct_pos, pct_neg, issues, prod[0])

    # Sample reviews
    def sample(sent: str, n: int) -> list[str]:
        return [texts[i] for i, s in enumerate(sentiments) if s == sent][:n]

    return {
        "url": product_url,
        "name": prod[0],
        "brand": prod[1] or "",
        "price": float(prod[2]) if prod[2] else 0,
        "rating": float(prod[3]) if prod[3] else 0,
        "reviewCount": int(prod[4]) if prod[4] else 0,
        "imageUrl": prod[5] or "",
        "sentiment": {
            "positive": pct_pos,
            "neutral": pct_neu,
            "negative": pct_neg,
        },
        "totalReviews": total,
        "positiveCount": pos,
        "neutralCount": neu,
        "negativeCount": neg,
        "topIssues": issues,
        "aiSummary": summary,
        "samples": {
            "positive": sample("positive", 2),
            "neutral": sample("neutral", 1),
            "negative": sample("negative", 2),
        },
    }


def refresh_cache() -> None:
    """Clear in-memory cache so next request re-queries DuckDB."""
    with _cache_lock:
        _cache.clear()
        _cache["_ts"] = 0.0


def health() -> dict:
    """Check DuckDB accessibility."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        n = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        con.close()
        return {"status": "ok", "reviews": n, "db_size_kb": DB_PATH.stat().st_size // 1024}
    except Exception as e:
        return {"status": "error", "error": str(e)}
