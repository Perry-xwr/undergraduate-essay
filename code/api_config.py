"""Read local API credentials without storing them in source control."""

import os


def get_deepseek_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("请先设置 DEEPSEEK_API_KEY 环境变量，再使用自然语言解析或报告功能。")
    return key
