"""Record the real public Streamlit UI and export an annotated MP4 demo.

Unlike the previous renderer, this script does not recreate the dashboard with
Pillow or Matplotlib. It starts ``web_public.py``, opens it in Chromium, clicks
the actual tabs, scrolls the actual page, and records the browser viewport.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import Locator, Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "demo" / "public_dashboard_demo.mp4"
DEFAULT_SCREENSHOT = ROOT / "docs" / "public-dashboard-overview.png"
VIEWPORT = {"width": 1440, "height": 900}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{url.rstrip('/')}/_stcore/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.read().decode("utf-8").strip() == "ok":
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(0.3)
    raise TimeoutError(f"Streamlit did not become ready: {health_url} ({last_error})")


def start_streamlit(port: int) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "web_public.py"),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_dashboard(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    page.get_by_role(
        "heading",
        name="学習行動 異常検知ダッシュボード",
        exact=True,
    ).wait_for(state="visible", timeout=120_000)
    page.get_by_text("対象学生", exact=True).first.wait_for(state="visible", timeout=120_000)
    page.wait_for_timeout(2_000)


def inject_recording_styles(page: Page) -> None:
    page.add_style_tag(
        content="""
        html { scroll-behavior: smooth !important; }
        * { caret-color: transparent !important; }
        [data-demo-highlight="true"] {
          outline: 4px solid #f7b32b !important;
          outline-offset: 4px !important;
          border-radius: 8px !important;
          box-shadow: 0 0 0 7px rgba(247, 179, 43, 0.18) !important;
        }
        """
    )


def show_caption(page: Page, title: str, body: str) -> None:
    page.evaluate(
        """
        ({ title, body }) => {
          let box = document.getElementById('public-demo-caption');
          if (!box) {
            box = document.createElement('div');
            box.id = 'public-demo-caption';
            Object.assign(box.style, {
              position: 'fixed',
              left: '352px',
              right: '28px',
              bottom: '24px',
              zIndex: '2147483647',
              pointerEvents: 'none',
              color: '#ffffff',
              background: 'rgba(15, 23, 42, 0.93)',
              border: '1px solid rgba(255, 255, 255, 0.20)',
              borderRadius: '14px',
              padding: '15px 20px 16px',
              boxShadow: '0 18px 50px rgba(15, 23, 42, 0.32)',
              fontFamily: 'sans-serif',
              lineHeight: '1.45'
            });
            document.body.appendChild(box);
          }
          box.innerHTML = '';
          const heading = document.createElement('div');
          heading.textContent = title;
          Object.assign(heading.style, {
            color: '#9ec1ff',
            fontWeight: '800',
            fontSize: '17px',
            marginBottom: '4px'
          });
          const description = document.createElement('div');
          description.textContent = body;
          Object.assign(description.style, {
            fontWeight: '600',
            fontSize: '15px',
            letterSpacing: '0.01em'
          });
          box.append(heading, description);
        }
        """,
        {"title": title, "body": body},
    )


def hide_caption(page: Page) -> None:
    page.evaluate("document.getElementById('public-demo-caption')?.remove()")


def clear_highlight(page: Page) -> None:
    page.locator('[data-demo-highlight="true"]').evaluate_all(
        "elements => elements.forEach(element => element.removeAttribute('data-demo-highlight'))"
    )


def highlight(page: Page, locator: Locator) -> None:
    clear_highlight(page)
    locator.first.evaluate("element => element.setAttribute('data-demo-highlight', 'true')")


def pause(page: Page, seconds: float) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def show_step(
    page: Page,
    locator: Locator,
    title: str,
    body: str,
    seconds: float,
    click: bool = False,
) -> None:
    locator.first.scroll_into_view_if_needed(timeout=30_000)
    pause(page, 0.8)
    if click:
        locator.first.click(timeout=30_000)
        pause(page, 1.4)
    highlight(page, locator)
    show_caption(page, title, body)
    pause(page, seconds)


def find_ffmpeg() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg was not found. Install requirements-video.txt or provide ffmpeg on PATH."
        ) from None


def convert_to_mp4(source: Path, output: Path, trim_start: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(trim_start),
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def record_demo(
    url: str,
    output: Path,
    screenshot: Path,
    browser_path: str | None,
    keep_webm: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    screenshot.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="public-dashboard-video-") as temp_dir:
        temp_path = Path(temp_dir)
        video_dir = temp_path / "recordings"
        video_dir.mkdir()

        with sync_playwright() as playwright:
            launch_options = {
                "headless": True,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if browser_path:
                launch_options["executable_path"] = browser_path
            browser = playwright.chromium.launch(**launch_options)
            context = browser.new_context(
                viewport=VIEWPORT,
                locale="ja-JP",
                color_scheme="light",
                record_video_dir=str(video_dir),
                record_video_size=VIEWPORT,
            )
            page = context.new_page()
            wait_for_dashboard(page, url)
            inject_recording_styles(page)

            hide_caption(page)
            clear_highlight(page)
            page.evaluate("window.scrollTo({ top: 0, behavior: 'instant' })")
            pause(page, 0.5)
            page.screenshot(path=str(screenshot), full_page=False)

            show_caption(
                page,
                "公開デモ",
                "この映像は、合成データで動作する実際のStreamlit画面をそのまま収録しています。",
            )
            pause(page, 4.0)

            sidebar = page.locator('[data-testid="stSidebar"]')
            highlight(page, sidebar)
            show_caption(
                page,
                "1. 分析条件",
                "初期状態では3コースと1〜8週目をすべて選択し、学生×週の単位で分析します。",
            )
            pause(page, 4.5)

            overview_tab = page.get_by_role("tab", name="概要", exact=True)
            show_step(
                page,
                overview_tab,
                "2. 概要",
                "灰色が通常範囲、赤が要確認です。点は1人の1週間分の行動を表します。",
                5.0,
                click=True,
            )

            ranking_tab = page.get_by_role("tab", name="要確認ランキング", exact=True)
            show_step(
                page,
                ranking_tab,
                "3. 要確認ランキング",
                "相対スコアの高い学生・週から確認対象を絞り、結果をCSVで保存できます。",
                5.5,
                click=True,
            )

            detail_tab = page.get_by_role("tab", name="学生・週の詳細", exact=True)
            show_step(
                page,
                detail_tab,
                "4. 学生・週の詳細",
                "選択した学生・週について、操作カテゴリの内訳と時系列ログを照合します。",
                6.0,
                click=True,
            )

            behavior_tab = page.get_by_role("tab", name="行動タイプ", exact=True)
            show_step(
                page,
                behavior_tab,
                "5. 行動タイプ",
                "似た行動をグループ化して比較します。タイプ番号に優劣の意味はありません。",
                6.0,
                click=True,
            )

            show_step(
                page,
                overview_tab,
                "まとめ",
                "全体分布から対象を絞り、ランキング、個別ログ、行動タイプの順に確認できます。",
                4.0,
                click=True,
            )
            clear_highlight(page)
            pause(page, 0.5)

            video = page.video
            page.close()
            context.close()
            raw_video = Path(video.path())
            browser.close()

        saved_webm = output.with_suffix(".webm")
        if keep_webm:
            shutil.copy2(raw_video, saved_webm)
        convert_to_mp4(raw_video, output, trim_start=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Record an already running public app instead of launching Streamlit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--browser-path", help="Optional Chromium/Chrome executable")
    parser.add_argument("--keep-webm", action="store_true")
    args = parser.parse_args()

    streamlit_process: subprocess.Popen | None = None
    try:
        if args.url:
            url = args.url.rstrip("/")
        else:
            port = free_port()
            url = f"http://127.0.0.1:{port}"
            streamlit_process = start_streamlit(port)
            wait_for_server(url)
        record_demo(
            url=url,
            output=args.output.resolve(),
            screenshot=args.screenshot.resolve(),
            browser_path=args.browser_path,
            keep_webm=args.keep_webm,
        )
    finally:
        stop_process(streamlit_process)

    print(f"Video: {args.output.resolve()}")
    print(f"Screenshot: {args.screenshot.resolve()}")


if __name__ == "__main__":
    main()
