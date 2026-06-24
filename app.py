#!/usr/bin/env python3
"""
Flask app — Trendyol Sentiment Analysis Dashboard.
Connects to DuckDB, serves live sentiment data + HTML dashboard.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request

# Ensure the engine module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_engine import get_products, get_products_with_sentiment, get_product_sentiment, refresh_cache, health

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # preserve Turkish/Azeri chars


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main dashboard."""
    return render_template("dashboard.html")


@app.route("/api/health")
def api_health():
    return jsonify(health())


@app.route("/api/products")
def api_products():
    """Return lightweight product list."""
    products = get_products()
    return jsonify({
        "products": products,
        "total": len(products),
    })


@app.route("/api/products/summary")
def api_products_summary():
    """Return products with mini sentiment data. Use ?limit=0 for all."""
    limit = request.args.get("limit", 200, type=int)  # default: all
    if limit < 1:
        limit = None
    products = get_products_with_sentiment(limit=limit)
    return jsonify({
        "products": products,
        "total": len(products),
    })


@app.route("/api/brands")
def api_brands():
    """Return distinct brands with product/review counts."""
    products = get_products()
    brand_map: dict[str, dict] = {}
    for p in products:
        b = p["brand"] or "Unknown"
        if b not in brand_map:
            brand_map[b] = {"brand": b, "productCount": 0}
        brand_map[b]["productCount"] += 1
    brands = sorted(brand_map.values(), key=lambda x: -x["productCount"])
    return jsonify({"brands": brands, "total": len(brands)})
def api_product():
    """Full sentiment analysis for one product (by ?url=...)."""
    url = request.args.get("url", "")
    if not url:
        return jsonify({"error": "Missing ?url= parameter"}), 400

    data = get_product_sentiment(url)
    if data is None:
        return jsonify({"error": "Product not found or has no reviews"}), 404
    return jsonify(data)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Clear the in-memory cache — next request re-queries DuckDB."""
    refresh_cache()
    return jsonify({"status": "ok", "message": "Cache cleared"})


# ── CLI entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    default_port = int(os.environ.get("PORT", 5999))
    port = default_port
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"🔌 DuckDB sentiment server → http://localhost:{port}")
    print(f"   Data: {health().get('reviews', '?')} reviews in DuckDB")
    print(f"   Dashboard: http://localhost:{port}/")
    print(f"   API:       http://localhost:{port}/api/products")
    app.run(host="0.0.0.0", port=port, debug=debug)
