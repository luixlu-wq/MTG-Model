from __future__ import annotations

import json


SYSTEM_PROMPT = (
    "You are an expert on Chinese 5A tourism. "
    "Provide accurate, grounded answers. "
    "Do not invent facts; if unsure, say you are unsure."
)


def build_prompt(spot: dict) -> list[dict]:
    name = spot.get("name") or ""
    prov = spot.get("province") or ""
    details = (spot.get("text_zh") or {}).get("details") or {}
    short = (spot.get("text_zh") or {}).get("short_intro") or ""
    prompt = (
        f"你是中国5A旅游专家。请严格输出一个JSON对象，不要输出多余文本。"
        f"景区：{prov}{name}。\n"
        f"JSON结构如下（字段可为空，但必须保留键）：\n"
        f"{{\n"
        f'  "name": "...",\n'
        f'  "province": "...",\n'
        f'  "history": ["..."],\n'
        f'  "culture": ["..."],\n'
        f'  "logistics": ["..."],\n'
        f'  "tickets_hours": ["..."],\n'
        f'  "photo_spots": [{{"theme": "...", "spot": "...", "note": "..."}}],\n'
        f'  "attractions": [{{"name": "...", "type": "...", "desc": "...", "lat": null, "lon": null}}],\n'
        f'  "routes": [{{"name": "...", "duration": "...", "steps": ["..."]}}],\n'
        f'  "myths": ["..."],\n'
        f'  "geo": {{"lat": null, "lon": null, "hint": "..."}}\n'
        f"}}\n"
        f"要求：\n"
        f"- 不要编造；未知请写“未知”。\n"
        f"- 如果已知经纬度，请填写geo.lat/geo.lon。\n"
        f"- attractions中的经纬度如未知可留null。\n"
        f"\n可用信息：{short}；细节：{json.dumps(details, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_itinerary_prompt(spot: dict) -> list[dict]:
    name = spot.get("name") or ""
    prov = spot.get("province") or ""
    prompt = (
        f"请严格输出一个JSON对象，不要输出多余文本。为{prov}{name}设计1日游线路，要求包含逐步导航。\n"
        f"JSON结构如下（字段可为空，但必须保留键）：\n"
        f"{{\n"
        f'  "name": "{prov}{name}",\n'
        f'  "route_name": "...",\n'
        f'  "total_duration": "...",\n'
        f'  "steps": [\n'
        f'    {{\n'
        f'      "from": "...",\n'
        f'      "to": "...",\n'
        f'      "direction": "...",\n'
        f'      "distance": "...",\n'
        f'      "duration": "...",\n'
        f'      "lat": null,\n'
        f'      "lon": null,\n'
        f'      "notes": "..." \n'
        f'    }}\n'
        f'  ],\n'
        f'  "photo_spots": [{{"theme": "...", "spot": "...", "time": "..."}}, ...],\n'
        f'  "tips": ["...", "..."],\n'
        f'  "guide_text": "..." \n'
        f"}}\n"
        f"要求：\n"
        f"- steps必须覆盖从一个点到下一个点的导航指引。\n"
        f"- 如果已知经纬度，请填写lat/lon，否则填null并在notes中说明参考地标。\n"
        f"- 不要编造；未知请写“未知”。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_bilingual_prompt(spot: dict) -> list[dict]:
    name = spot.get("name") or ""
    prov = spot.get("province") or ""
    prompt = (
        f"请分别用中文和英文介绍{prov}{name}（先中文后英文），"
        f"包含历史、文化、交通、最佳游览时间与简短游览建议。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
