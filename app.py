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
    """Return top N products with mini sentiment data (for grid view)."""
    limit = request.args.get("limit", 20, type=int)
    products = get_products_with_sentiment(limit=limit)
    return jsonify({
        "products": products,
        "total": len(products),
    })


@app.route("/api/product")
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
