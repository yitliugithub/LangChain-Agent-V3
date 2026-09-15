"""Small, user-triggered Douyin Index collection experiment.

The script opens a dedicated Chrome profile. On the first run, log in manually in
that window and press Enter in the terminal. It does not store account passwords,
bypass verification, or call undocumented Douyin APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_DIR = PROJECT_DIR / "storage/douyin_browser_profile"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/douyin"
DOUYIN_INDEX_URL = (
    "https://creator.douyin.com/creator-micro/creator-count/"
    "arithmetic-index/analysis"
)
TAB_NAMES = {
    "heat_index": "keyword_index",
    "correlation": "correlation",
    "crowd": "audience",
}
TIME_RANGE_LABELS = {
    "7d": "过去7天",
    "14d": "过去14天",
    "30d": "过去30天",
    "6m": "过去半年",
}
CONFIGURABLE_PERIOD_TABS = {"heat_index", "crowd"}
DATE_PERIOD_PATTERN = re.compile(
    r"(20\d{2}-\d{2}-\d{2})\s*[~～-]+\s*(20\d{2}-\d{2}-\d{2})"
)
SCROLL_STEP_RATIO = 0.82
SCREENSHOT_OVERLAP = 120
NOT_INDEXED_MARKERS = (
    "暂未收录该关键词",
    "关键词尚未收录",
    "尚未收录该关键词",
    "创建新词能力升级中",
)
MAX_APPROVED_KEYWORDS = 10


class KeywordNotIndexedError(RuntimeError):
    """Raised when Douyin Index explicitly reports that a keyword is absent."""


@dataclass(frozen=True)
class KeywordCandidate:
    keyword: str
    keyword_type: str
    source: str
    required: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Douyin Index keywords in a persistent browser session."
    )
    parser.add_argument("--keyword", help="Keyword to collect, for example: 苹果")
    parser.add_argument(
        "--brief",
        type=Path,
        help="Optional JSON research brief used to create a keyword plan.",
    )
    parser.add_argument(
        "--keyword-index",
        type=int,
        default=0,
        help="Index in the generated keyword plan; default: 0.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index when --brief JSON contains multiple tasks; default: 0.",
    )
    parser.add_argument(
        "--all-approved",
        action="store_true",
        help="Collect every approved keyword from --brief in one browser session.",
    )
    parser.add_argument(
        "--time-range",
        choices=tuple(TIME_RANGE_LABELS),
        help=(
            "关键词指数和人群分析的周期。单关键词模式默认30d；"
            "brief模式默认读取preferred_time_range。"
        ),
    )
    parser.add_argument(
        "--batch-number",
        type=int,
        help="Collect one 1-based batch from the approved keyword plan.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=3,
        help="Number of stable keyword batches; default: 3.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
    )
    parser.add_argument(
        "--login-timeout",
        type=int,
        default=300,
        help="Seconds to wait for manual login; default: 300.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Only use after the persistent profile has a valid login session.",
    )
    return parser.parse_args()


def clean_keyword(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def split_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"[,，、;；/\n]", str(value or ""))
    return [keyword for item in raw_values if (keyword := clean_keyword(item))]


def build_keyword_plan(brief: dict[str, Any]) -> list[KeywordCandidate]:
    """Build deterministic required keywords before any optional LLM expansion."""
    if "approved_keywords" in brief:
        candidates: list[KeywordCandidate] = []
        seen: set[str] = set()
        for item in brief.get("approved_keywords", []):
            keyword = clean_keyword(item.get("keyword"))
            normalized = keyword.casefold()
            if not keyword or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                KeywordCandidate(
                    keyword=keyword,
                    keyword_type=clean_keyword(item.get("keyword_type")) or "approved",
                    source=f"keyword_review.{clean_keyword(item.get('source_field'))}",
                    required=clean_keyword(item.get("required")).casefold() == "yes",
                )
            )
        return candidates

    field_map = (
        ("brand_name", "brand", True),
        ("brand_aliases", "brand_alias", True),
        ("product_name", "product", True),
        ("product_category", "category", True),
        ("required_keywords", "required", True),
        ("parent_category", "parent_category", False),
        ("competitor_brands", "competitor_brand", False),
        ("competitor_products", "competitor_product", False),
        ("selling_points", "selling_point", False),
        ("ingredient_keywords", "ingredient", False),
    )
    excluded = set(split_keywords(brief.get("excluded_keywords")))
    candidates: list[KeywordCandidate] = []
    seen: set[str] = set()

    for field_name, keyword_type, required in field_map:
        for keyword in split_keywords(brief.get(field_name)):
            normalized = keyword.casefold()
            if keyword in excluded or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                KeywordCandidate(
                    keyword=keyword,
                    keyword_type=keyword_type,
                    source=f"input_table.{field_name}",
                    required=required,
                )
            )
    return candidates


def load_keywords(
    args: argparse.Namespace,
) -> tuple[list[KeywordCandidate], list[dict[str, Any]]]:
    if args.all_approved and args.keyword:
        raise ValueError("--all-approved 只能与 --brief 一起使用")
    if args.keyword:
        keyword = clean_keyword(args.keyword)
        if not keyword:
            raise ValueError("--keyword 不能为空")
        candidate = KeywordCandidate(keyword, "manual", "cli", True)
        return [candidate], [asdict(candidate)]

    if not args.brief:
        raise ValueError("请提供 --keyword 或 --brief")
    with args.brief.open("r", encoding="utf-8") as file:
        brief = json.load(file)
    if isinstance(brief, dict) and "tasks" in brief:
        tasks = brief.get("tasks") or []
        if not 0 <= args.task_index < len(tasks):
            raise IndexError(
                f"--task-index 超出范围；当前输入共有 {len(tasks)} 个任务"
            )
        brief = tasks[args.task_index]
    plan = build_keyword_plan(brief)
    if not plan:
        raise ValueError("输入表没有生成任何可查询关键词")

    if args.all_approved:
        if args.batch_number is not None:
            raise ValueError("--all-approved 不能与 --batch-number 同时使用")
        if len(plan) > MAX_APPROVED_KEYWORDS:
            raise ValueError(
                f"一次最多采集 {MAX_APPROVED_KEYWORDS} 个 approved 关键词；"
                f"当前共有 {len(plan)} 个"
            )
        return plan, [asdict(item) for item in plan]

    if args.batch_number is not None:
        if args.batch_count < 1:
            raise ValueError("--batch-count 必须大于 0")
        if not 1 <= args.batch_number <= args.batch_count:
            raise ValueError("--batch-number 必须位于 1 和 --batch-count 之间")
        start = len(plan) * (args.batch_number - 1) // args.batch_count
        end = len(plan) * args.batch_number // args.batch_count
        selected = plan[start:end]
        if not selected:
            raise ValueError(
                f"第 {args.batch_number} 批没有关键词；当前共有 {len(plan)} 个词"
            )
        return selected, [asdict(item) for item in plan]

    if not 0 <= args.keyword_index < len(plan):
        raise IndexError(
            f"--keyword-index 超出范围；当前关键词计划共有 {len(plan)} 项"
        )
    return [plan[args.keyword_index]], [asdict(item) for item in plan]


def load_keyword(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    """Backward-compatible single-keyword loader."""
    selected, plan = load_keywords(args)
    return selected[0].keyword, plan


def resolve_time_range(args: argparse.Namespace) -> str:
    if args.time_range:
        return args.time_range
    if not args.brief:
        return "30d"

    with args.brief.open("r", encoding="utf-8") as file:
        brief = json.load(file)
    if isinstance(brief, dict) and "tasks" in brief:
        tasks = brief.get("tasks") or []
        if not 0 <= args.task_index < len(tasks):
            raise IndexError(
                f"--task-index 超出范围；当前输入共有 {len(tasks)} 个任务"
            )
        brief = tasks[args.task_index]

    requested = clean_keyword(brief.get("preferred_time_range"))
    if requested not in TIME_RANGE_LABELS:
        raise ValueError(
            "preferred_time_range 必须是 7d、14d、30d 或 6m"
        )
    return requested


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w\-.\u4e00-\u9fff]+", "_", value).strip("_")
    return cleaned or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def build_url(keyword: str, tab: str) -> str:
    return (
        f"{DOUYIN_INDEX_URL}?keyword={quote(keyword)}&tab={tab}"
        "&appName=aweme&source=creator"
    )


def navigate(page: Page, url: str) -> None:
    """Navigate without waiting for analytics requests that may remain open."""
    if page.url == url:
        return
    try:
        page.goto(url, wait_until="commit", timeout=30_000)
    except PlaywrightTimeoutError:
        if page.url != url:
            raise


def page_text(page: Page) -> str:
    return page.locator("body").inner_text(timeout=10_000)


def page_text_with_inputs(page: Page) -> str:
    text = page_text(page)
    input_values = page.locator("input:visible").evaluate_all(
        "elements => elements.map(element => element.value).filter(Boolean)"
    )
    return text + "\n" + "\n".join(input_values)


def extract_date_periods(text: str) -> list[tuple[str, str]]:
    return list(dict.fromkeys(DATE_PERIOD_PATTERN.findall(text)))


def period_matches(period: tuple[str, str], requested: str) -> bool:
    start = datetime.strptime(period[0], "%Y-%m-%d").date()
    end = datetime.strptime(period[1], "%Y-%m-%d").date()
    days = (end - start).days
    expected_ranges = {
        "7d": (5, 9),
        "14d": (12, 16),
        "30d": (27, 33),
        "6m": (170, 190),
    }
    minimum, maximum = expected_ranges[requested]
    return minimum <= days <= maximum


def click_first_visible(locator) -> bool:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            candidate.click()
            return True
    return False


def open_period_picker(page: Page) -> None:
    date_inputs = page.locator("input:visible")
    for index in range(date_inputs.count()):
        candidate = date_inputs.nth(index)
        value = candidate.input_value().strip()
        # Different tabs render either one range input or two date inputs.
        if DATE_PERIOD_PATTERN.search(value) or re.fullmatch(
            r"20\d{2}-\d{2}-\d{2}", value
        ):
            # ByteDance's date input is readonly and covered by its label wrapper.
            candidate.evaluate(
                "element => (element.closest('label') || element).click()"
            )
            return

    date_text = page.get_by_text(
        re.compile(
            r"20\d{2}-\d{2}-\d{2}\s*[~～-]+\s*20\d{2}-\d{2}-\d{2}"
        )
    )
    if click_first_visible(date_text):
        return
    raise RuntimeError("找不到可点击的时间周期控件")


def configure_period(page: Page, tab: str, requested: str) -> dict[str, Any]:
    if tab == "correlation":
        periods = extract_date_periods(page_text_with_inputs(page))
        actual = periods[0] if periods else None
        return {
            "requested_period": requested,
            "selected_period": "7d",
            "actual_date_range": (
                f"{actual[0]} ~ {actual[1]}" if actual else None
            ),
            "period_policy": "platform_fixed_7d",
            "period_verified": bool(actual and period_matches(actual, "7d")),
        }

    if tab not in CONFIGURABLE_PERIOD_TABS:
        raise ValueError(f"未知的时间周期模块：{tab}")

    periods = extract_date_periods(page_text_with_inputs(page))
    if periods and period_matches(periods[0], requested):
        actual = periods[0]
    else:
        open_period_picker(page)
        label = TIME_RANGE_LABELS[requested]
        # The popup is animated and some versions insert whitespace in labels.
        option_pattern = re.compile(
            rf"过去\s*{requested[:-1]}\s*天"
            if requested.endswith("d")
            else r"过去\s*半年"
        )
        option = page.get_by_text(option_pattern)
        option_deadline = time.monotonic() + 5
        while time.monotonic() < option_deadline:
            if click_first_visible(option):
                break
            page.wait_for_timeout(250)
        else:
            raise RuntimeError(f"时间控件中找不到选项：{label}")

        deadline = time.monotonic() + 30
        actual = None
        while time.monotonic() < deadline:
            page.wait_for_timeout(800)
            current_periods = extract_date_periods(page_text_with_inputs(page))
            if current_periods and period_matches(current_periods[0], requested):
                actual = current_periods[0]
                break
        if actual is None:
            raise TimeoutError(f"选择{label}后，页面时间范围未通过验证")

    return {
        "requested_period": requested,
        "selected_period": requested,
        "actual_date_range": f"{actual[0]} ~ {actual[1]}",
        "period_policy": "configured",
        "period_verified": True,
    }


def is_index_page_ready(page: Page) -> bool:
    try:
        text = page_text(page)
    except PlaywrightTimeoutError:
        return False
    return (
        "抖音指数" in text and "关键词指数" in text
    ) or any(marker in text for marker in NOT_INDEXED_MARKERS)


def wait_for_manual_login(page: Page, timeout_seconds: int) -> None:
    # A persisted session still needs a few seconds to hydrate the analytics UI.
    for _ in range(25):
        if is_index_page_ready(page):
            return
        time.sleep(1)

    print("\n[Login] 请在打开的 Chrome 窗口中手动登录抖音创作者中心。")
    print("[Login] 登录完成并看到‘抖音指数’页面后，回到终端按 Enter。")
    input()

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_index_page_ready(page):
            return
        time.sleep(1)
    raise TimeoutError("等待抖音创作者中心登录超时")


def get_scroll_container(page: Page):
    candidates = (
        ".scroll-container",
        "[class*='scroll-container']",
        "main",
    )
    for selector in candidates:
        locator = page.locator(selector).first
        if locator.count() and locator.is_visible():
            metrics = locator.evaluate(
                "el => ({height: el.clientHeight, scrollHeight: el.scrollHeight})"
            )
            if metrics["scrollHeight"] > metrics["height"] + 50:
                return locator
    return page.locator("html")


def wait_until_stable(page: Page, keyword: str, timeout_seconds: int = 35) -> str:
    """Wait for async content and require two consecutive stable text snapshots."""
    deadline = time.monotonic() + timeout_seconds
    previous_digest = None
    stable_count = 0
    last_text = ""

    while time.monotonic() < deadline:
        last_text = page_text(page)
        if any(marker in last_text for marker in NOT_INDEXED_MARKERS):
            raise KeywordNotIndexedError(f"抖音指数暂未收录关键词：{keyword}")
        has_keyword = keyword in last_text
        has_content = any(
            marker in last_text
            for marker in ("平均值", "关联词排名", "地域分布")
        )
        digest = hashlib.sha1(last_text.encode("utf-8")).hexdigest()
        if has_keyword and has_content and digest == previous_digest:
            stable_count += 1
            if stable_count >= 2:
                return last_text
        else:
            stable_count = 0
        previous_digest = digest
        time.sleep(1)

    if keyword not in last_text:
        raise TimeoutError(f"页面没有稳定切换到关键词：{keyword}")
    return last_text


def scroll_to_load_all(page: Page) -> tuple[Any, dict[str, int]]:
    scroll_area = get_scroll_container(page)
    metrics = scroll_area.evaluate(
        "el => ({height: el.clientHeight, scrollHeight: el.scrollHeight})"
    )
    height = max(int(metrics["height"]), 1)
    total = max(int(metrics["scrollHeight"]), height)
    step = max(int(height * SCROLL_STEP_RATIO), 300)

    for position in range(0, total, step):
        scroll_area.evaluate("(el, y) => { el.scrollTop = y; }", position)
        page.wait_for_timeout(650)
    scroll_area.evaluate("el => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(1000)
    metrics = scroll_area.evaluate(
        "el => ({height: el.clientHeight, scrollHeight: el.scrollHeight})"
    )
    return scroll_area, {
        "height": int(metrics["height"]),
        "scroll_height": int(metrics["scrollHeight"]),
    }


def capture_scrolling_screenshot(
    page: Page,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    """Capture overlapping views of the inner scroller and stitch one long image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scroll_area, metrics = scroll_to_load_all(page)
    box = scroll_area.bounding_box()
    if not box:
        raise RuntimeError("无法确定页面滚动区域")

    viewport = page.viewport_size or {"width": 1440, "height": 1000}
    clip_x = max(int(box["x"]), 0)
    clip_y = max(int(box["y"]), 0)
    clip_width = min(int(box["width"]), viewport["width"] - clip_x)
    clip_height = min(int(box["height"]), viewport["height"] - clip_y)
    if clip_width <= 0 or clip_height <= 0:
        raise RuntimeError("滚动区域不在当前浏览器视口内")

    max_scroll = max(metrics["scroll_height"] - metrics["height"], 0)
    step = max(clip_height - SCREENSHOT_OVERLAP, 200)
    positions = list(range(0, max_scroll + 1, step))
    if not positions or positions[-1] != max_scroll:
        positions.append(max_scroll)

    segment_paths: list[Path] = []
    for index, position in enumerate(positions, start=1):
        scroll_area.evaluate("(el, y) => { el.scrollTop = y; }", position)
        page.wait_for_timeout(700)
        segment_path = output_dir / f"{prefix}_segment_{index:02d}.png"
        page.screenshot(
            path=str(segment_path),
            clip={
                "x": clip_x,
                "y": clip_y,
                "width": clip_width,
                "height": clip_height,
            },
        )
        segment_paths.append(segment_path)

    stitched_path = output_dir / f"{prefix}_stitched_fallback.png"
    images = [Image.open(path).convert("RGB") for path in segment_paths]
    try:
        canvas_height = positions[-1] + images[-1].height
        canvas = Image.new("RGB", (images[0].width, canvas_height), "white")
        for position, screenshot in zip(positions, images):
            canvas.paste(screenshot, (0, position))
        canvas.save(stitched_path, quality=92)
    finally:
        for screenshot in images:
            screenshot.close()

    scroll_area.evaluate("el => { el.scrollTop = 0; }")
    full_path = output_dir / f"{prefix}_full.png"
    try:
        scroll_area.evaluate(
            """el => {
                el.scrollTop = 0;
                el.style.setProperty('height', `${el.scrollHeight}px`, 'important');
                el.style.setProperty('max-height', 'none', 'important');
                el.style.setProperty('overflow', 'visible', 'important');
            }"""
        )
        page.wait_for_timeout(500)
        scroll_area.screenshot(path=str(full_path), timeout=30_000)
    except Exception:
        Image.open(stitched_path).save(full_path)

    return {
        "full_screenshot": str(full_path),
        "stitched_fallback": str(stitched_path),
        "segments": [str(path) for path in segment_paths],
        "scroll_height": metrics["scroll_height"],
    }


def parse_common(text: str, keyword: str, url: str) -> dict[str, Any]:
    unique_periods = extract_date_periods(text)
    return {
        "keyword": keyword,
        "source_url": url,
        "data_periods": [f"{start} ~ {end}" for start, end in unique_periods],
    }


def parse_keyword_index(text: str, keyword: str, url: str) -> dict[str, Any]:
    result = parse_common(text, keyword, url)
    summaries = re.findall(
        r"同比\s*([+\-]?\d+(?:\.\d+)?%)\s*[｜|]\s*环比\s*"
        r"([+\-]?\d+(?:\.\d+)?%)\s*平均值\s*([\d.]+万?)",
        text,
    )
    unique_summaries = []
    seen = set()
    for yoy, mom, average in summaries:
        key = (yoy, mom, average)
        if key in seen:
            continue
        seen.add(key)
        unique_summaries.append(
            {"year_on_year": yoy, "month_on_month": mom, "average": average}
        )
    result["index_summaries"] = unique_summaries
    return result


def parse_correlation(text: str, keyword: str, url: str) -> dict[str, Any]:
    result = parse_common(text, keyword, url)
    result["has_data"] = "暂无数据" not in text
    compact = re.sub(r"\s+", "", text)
    summary_match = re.search(
        rf"搜索{re.escape(keyword)}的人也都在搜([^，,]+)[，,]其中([^，,]+)最近搜索飙升",
        compact,
    )
    result["summary"] = (
        {
            "also_searched": summary_match.group(1),
            "rising_search": summary_match.group(2),
        }
        if summary_match
        else None
    )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ranking = []
    try:
        start = lines.index("按涨幅") + 1
        cursor = start
        for expected_rank in range(1, 21):
            if expected_rank >= 4 and cursor < len(lines):
                if lines[cursor] == str(expected_rank):
                    cursor += 1
            if cursor + 1 >= len(lines):
                break
            term, score = lines[cursor], lines[cursor + 1]
            if not re.fullmatch(r"\d+(?:\.\d+)?", score):
                break
            ranking.append(
                {"rank": expected_rank, "term": term, "score": float(score)}
            )
            cursor += 2
    except ValueError:
        pass
    result["ranking_by_relevance"] = ranking
    return result


def parse_audience(text: str, keyword: str, url: str) -> dict[str, Any]:
    result = parse_common(text, keyword, url)
    province_rows = re.findall(
        r"(?:^|\n)\s*(\d{1,2})\s+([\u4e00-\u9fff]{2,6})\s+"
        r"([\d.]+)%\s+([\d.]+)",
        text,
    )
    result["provinces"] = [
        {
            "rank": int(rank),
            "province": province,
            "share_percent": float(share),
            "tgi": float(tgi),
        }
        for rank, province, share, tgi in province_rows
    ]

    gender_rows = re.findall(
        r"(男性|女性)\s*\n?\s*占比\s*(\d+(?:\.\d+)?)%\s*TGI\s*(\d+(?:\.\d+)?)",
        text,
    )
    result["gender"] = [
        {"gender": gender, "share_percent": float(share), "tgi": float(tgi)}
        for gender, share, tgi in gender_rows
    ]

    age_summary = re.search(
        r"(\d{2}-\d{2}岁|\d{2}岁\+)\s*年龄段占比最高，\s*"
        r"(\d{2}-\d{2}岁|\d{2}岁\+)\s*年龄段偏好度",
        text,
    )
    result["age_summary"] = (
        {
            "highest_share": age_summary.group(1),
            "highest_tgi": age_summary.group(2),
        }
        if age_summary
        else None
    )

    interest_summary = re.search(
        r"([^\n]+兴趣)\s*的人群占比最高，\s*([^\n]+兴趣)\s*的人群偏好度",
        text,
    )
    result["interest_summary"] = (
        {
            "highest_share": interest_summary.group(1).strip(),
            "highest_tgi": interest_summary.group(2).strip(),
        }
        if interest_summary
        else None
    )
    return result


def collect_tab(
    page: Page,
    keyword: str,
    tab: str,
    run_dir: Path,
    requested_period: str,
) -> dict[str, Any]:
    name = TAB_NAMES[tab]
    url = build_url(keyword, tab)
    print(f"[Collect] {name}: {url}")
    navigate(page, url)
    wait_until_stable(page, keyword)
    period_info = configure_period(page, tab, requested_period)
    text = wait_until_stable(page, keyword)
    screenshot_info = capture_scrolling_screenshot(
        page,
        run_dir / "screenshots",
        name,
    )
    text = page_text(page)
    input_values = page.locator("input:visible").evaluate_all(
        "elements => elements.map(element => element.value).filter(Boolean)"
    )
    parse_text = text + "\n" + "\n".join(input_values)
    (run_dir / f"{name}_raw_text.txt").write_text(text, encoding="utf-8")

    parsers = {
        "heat_index": parse_keyword_index,
        "correlation": parse_correlation,
        "crowd": parse_audience,
    }
    data = parsers[tab](parse_text, keyword, page.url)
    data.update(
        {
            "tab": tab,
            "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "screenshots": screenshot_info,
            "status": "success",
            **period_info,
        }
    )
    with (run_dir / f"{name}.json").open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return data


def collect_keyword(
    page: Page,
    candidate: KeywordCandidate,
    keyword_plan: list[dict[str, Any]],
    output_root: Path,
    run_date: str,
    requested_period: str,
) -> dict[str, Any]:
    keyword = candidate.keyword
    run_dir = output_root / safe_name(keyword) / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "keyword": keyword,
        "keyword_candidate": asdict(candidate),
        "keyword_plan": keyword_plan,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "requested_period": requested_period,
        "tabs": {},
        "output_dir": str(run_dir.resolve()),
    }
    status_path = run_dir / "collection_status.json"

    try:
        for tab in TAB_NAMES:
            try:
                status["tabs"][tab] = collect_tab(
                    page,
                    keyword,
                    tab,
                    run_dir,
                    requested_period,
                )
            except KeywordNotIndexedError as exc:
                raw_text = page_text(page)
                (run_dir / "not_indexed_raw_text.txt").write_text(
                    raw_text,
                    encoding="utf-8",
                )
                evidence_path = run_dir / "not_indexed.png"
                page.screenshot(path=str(evidence_path), full_page=False)
                status["tabs"][tab] = {
                    "status": "skipped",
                    "reason": "keyword_not_indexed",
                    "error": str(exc),
                }
                for remaining_tab in list(TAB_NAMES)[list(TAB_NAMES).index(tab) + 1:]:
                    status["tabs"][remaining_tab] = {
                        "status": "skipped",
                        "reason": "keyword_not_indexed",
                    }
                status["status"] = "skipped"
                status["skip_reason"] = "keyword_not_indexed"
                status["evidence_screenshot"] = str(evidence_path.resolve())
                print(f"[Skip] {keyword}: 抖音指数暂未收录，进入下一个关键词")
                break
            except Exception as exc:
                status["tabs"][tab] = {
                    "status": "failed",
                    "error": str(exc),
                }
                print(f"[Failed] {keyword} / {TAB_NAMES[tab]}: {exc}")

        if status["status"] != "skipped":
            succeeded = sum(
                item.get("status") == "success" for item in status["tabs"].values()
            )
            status["status"] = "success" if succeeded == len(TAB_NAMES) else "partial"
            status["succeeded_tabs"] = succeeded
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
    finally:
        status["finished_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        with status_path.open("w", encoding="utf-8") as file:
            json.dump(status, file, ensure_ascii=False, indent=2)
    return status


def main() -> None:
    args = parse_args()
    selected, keyword_plan = load_keywords(args)
    requested_period = resolve_time_range(args)
    run_date = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    batch_name = (
        "all_approved"
        if args.all_approved
        else (
            f"batch_{args.batch_number}_of_{args.batch_count}"
            if args.batch_number is not None
            else "single_keyword"
        )
    )
    batch_dir = args.output_root / "_batches" / f"{run_date}_{batch_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_status: dict[str, Any] = {
        "batch_number": args.batch_number,
        "batch_count": args.batch_count if args.batch_number is not None else None,
        "keywords": [item.keyword for item in selected],
        "keyword_plan": keyword_plan,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "requested_period": requested_period,
        "results": [],
    }
    batch_status_path = batch_dir / "batch_status.json"

    print(
        f"[Batch] {batch_name}: "
        + "、".join(item.keyword for item in selected)
    )

    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(args.profile_dir),
                channel="chrome",
                headless=args.headless,
                viewport={"width": 1440, "height": 1000},
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = context.pages[0] if context.pages else context.new_page()
            navigate(page, build_url(selected[0].keyword, "heat_index"))
            wait_for_manual_login(page, args.login_timeout)

            for index, candidate in enumerate(selected, start=1):
                print(
                    f"\n[Keyword {index}/{len(selected)}] {candidate.keyword}"
                )
                result = collect_keyword(
                    page,
                    candidate,
                    keyword_plan,
                    args.output_root,
                    run_date,
                    requested_period,
                )
                batch_status["results"].append(result)

            context.close()

        succeeded = sum(
            item.get("status") == "success" for item in batch_status["results"]
        )
        skipped = sum(
            item.get("status") == "skipped" for item in batch_status["results"]
        )
        completed = succeeded + skipped
        if succeeded == len(selected):
            batch_status["status"] = "success"
        elif completed == len(selected):
            batch_status["status"] = "completed_with_skips"
        else:
            batch_status["status"] = "partial"
        batch_status["succeeded_keywords"] = succeeded
        batch_status["skipped_keywords"] = skipped
    except Exception as exc:
        batch_status["status"] = "failed"
        batch_status["error"] = str(exc)
        raise
    finally:
        batch_status["finished_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        with batch_status_path.open("w", encoding="utf-8") as file:
            json.dump(batch_status, file, ensure_ascii=False, indent=2)

    print(f"\n[Done] 批次状态：{batch_status_path}")


if __name__ == "__main__":
    main()
