import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency at runtime
    pdfplumber = None


DEFAULT_URL = "https://www.tunisievaleurs.com/nos-publications/notes-de-recherche/"
NOTES_FEED_URL = "https://data.tunisievaleurs.com/data/ww2_rech.aspx?t=F&l=F&s=200"
NOTES_DETAILS_URL = "https://data.tunisievaleurs.com/ww2_showdetart.aspx?artid={artid}"
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")


def normalize_text(value):
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.lower().strip()


def parse_date(text):
    match = DATE_RE.search(text or "")
    if not match:
        return None, None
    date_str = match.group(1)
    return dt.datetime.strptime(date_str, "%d/%m/%Y").date(), date_str


def parse_rfc_date(text):
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        return None


def count_dates(text):
    return len(DATE_RE.findall(text or ""))


def fetch_html(url, *, use_selenium=False, timeout=30):
    if use_selenium:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except Exception as exc:
            raise RuntimeError("Selenium requested but not available.") from exc
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            time.sleep(2)
            return driver.page_source
        finally:
            driver.quit()
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; notes-scraper/1.0)"}
    )
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def find_notes_section(soup):
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        label = normalize_text(header.get_text(" ", strip=True))
        if "notes de recherche" in label:
            return header.find_parent(["section", "div"]) or header.parent
    return soup


def find_card_candidates(root):
    candidates = []
    seen = set()
    for text_node in root.find_all(string=DATE_RE):
        current = text_node.parent
        best = None
        for _ in range(6):
            if current is None:
                break
            text = current.get_text(" ", strip=True)
            if count_dates(text) == 1 and current.find("a", href=True):
                best = current
            if count_dates(text) > 1:
                break
            current = current.parent
        if best is None:
            continue
        signature = normalize_text(best.get_text(" ", strip=True))[:120]
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(best)
    return candidates


def pick_title(card):
    for tag in card.find_all(["h1", "h2", "h3", "h4"]):
        text = tag.get_text(" ", strip=True)
        if text:
            return text
    for tag in card.find_all(["strong", "b"]):
        text = tag.get_text(" ", strip=True)
        if text:
            return text
    return None


def pick_details_link(card):
    for anchor in card.find_all("a", href=True):
        label = normalize_text(anchor.get_text(" ", strip=True))
        if "plus de details" in label or label == "details":
            return anchor.get("href")
    for anchor in card.find_all("a", href=True):
        href = anchor.get("href", "")
        if href and not href.startswith("#") and "javascript" not in href.lower():
            return href
    return None


def find_pdf_link(details_url, timeout=30):
    html = fetch_html(details_url, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        if ".pdf" in href.lower():
            return urljoin(details_url, href)
    for tag in soup.find_all(attrs=True):
        for value in tag.attrs.values():
            if isinstance(value, str) and ".pdf" in value.lower():
                return urljoin(details_url, value)
    for anchor in soup.find_all("a", href=True):
        label = normalize_text(anchor.get_text(" ", strip=True))
        if "telecharg" in label or "download" in label:
            return urljoin(details_url, anchor.get("href", ""))
    return None


def get_text_or_none(tag):
    if tag is None:
        return None
    return tag.get_text(strip=True)


def xml_text(element, tag):
    if element is None:
        return None
    node = element.find(tag)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def extract_artid(link):
    if not link:
        return None
    try:
        parsed = urlparse(link)
        params = parse_qs(parsed.query)
        artid = params.get("artid")
        if artid:
            return artid[0]
    except Exception:
        return None
    return None


def fetch_note_details(session, artid, timeout=30):
    if not artid:
        return None, None
    url = NOTES_DETAILS_URL.format(artid=artid)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    link = xml_text(root, "link")
    pub_date = xml_text(root, "pubDate")
    return link, pub_date


def scrape_notes_api(min_date, timeout=30):
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; notes-scraper/1.0)"}
    )
    response = session.get(NOTES_FEED_URL, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    items = []
    for item in root.findall(".//item"):
        title = xml_text(item, "title")
        pub_date = xml_text(item, "pubDate")
        link = xml_text(item, "link")
        date_obj = parse_rfc_date(pub_date)
        artid = extract_artid(link)
        pdf_link = None
        detail_date = None
        if artid:
            try:
                pdf_link, detail_date = fetch_note_details(
                    session, artid, timeout=timeout
                )
            except Exception:
                pdf_link = None
        detail_obj, _ = parse_date(detail_date or "")
        date_obj = detail_obj or date_obj
        if not date_obj:
            continue
        if date_obj < min_date:
            continue
        items.append(
            {
                "title": title,
                "date": date_obj.isoformat(),
                "pdf_link": pdf_link,
                "details_link": link,
            }
        )
    items.sort(key=lambda item: item["date"])
    return items


def scrape_notes(url, min_date, *, use_selenium=False, timeout=30):
    html = fetch_html(url, use_selenium=use_selenium, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")
    root = find_notes_section(soup)
    candidates = find_card_candidates(root)
    items = []
    seen = set()
    for card in candidates:
        text = card.get_text(" ", strip=True)
        date_obj, _ = parse_date(text)
        if not date_obj:
            continue
        if date_obj < min_date:
            continue
        title = pick_title(card)
        details_link = pick_details_link(card)
        if not details_link:
            continue
        details_url = urljoin(url, details_link)
        pdf_link = None
        if details_link.lower().endswith(".pdf"):
            pdf_link = details_url
        else:
            try:
                pdf_link = find_pdf_link(details_url, timeout=timeout)
            except Exception:
                pdf_link = None
        if not title:
            title = DATE_RE.sub("", text).strip()
        key = (title or "", date_obj.isoformat(), pdf_link or details_url)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "title": title,
                "date": date_obj.isoformat(),
                "pdf_link": pdf_link,
                "details_link": details_url,
            }
        )
    items.sort(key=lambda item: item["date"])
    if items:
        return items
    return scrape_notes_api(min_date, timeout=timeout)


def safe_filename(text, default="document.pdf"):
    if not text:
        return default
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    name = "".join(keep).strip("_")
    return name[:120] or default


def download_pdf(url, output_dir, *, timeout=60):
    if not url:
        return None
    os.makedirs(output_dir, exist_ok=True)
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or "report.pdf"
    filename = safe_filename(filename)
    dest = os.path.join(output_dir, filename)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    with open(dest, "wb") as handle:
        handle.write(response.content)
    return dest


def analyze_pdf(path):
    if fitz is None or pdfplumber is None:
        raise RuntimeError("PyMuPDF and pdfplumber are required for analysis.")

    doc = fitz.open(path)
    font_sizes = []
    for page in doc:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        font_sizes.append(span.get("size", 0))
    median_size = statistics.median(font_sizes) if font_sizes else 0

    header_candidates = []
    paragraphs = []
    for page in doc:
        page_text = page.get_text("text")
        for para in [p.strip() for p in page_text.split("\n\n") if p.strip()]:
            paragraphs.append(para)
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                texts = [span.get("text", "").strip() for span in line.get("spans", [])]
                text = " ".join([t for t in texts if t])
                if not text:
                    continue
                max_size = max(span.get("size", 0) for span in line.get("spans", []))
                if max_size >= median_size + 2:
                    header_candidates.append(text)
                elif text.isupper() and len(text.split()) <= 10:
                    header_candidates.append(text)

    unique_headers = []
    seen = set()
    for header in header_candidates:
        key = header.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_headers.append(header)

    word_counts = [len(p.split()) for p in paragraphs]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    sample_paragraphs = paragraphs[:5]

    text_tables = 0
    image_table_pages = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            if any(len(table) > 1 for table in tables):
                text_tables += len(tables)
                continue
            has_images = bool(page.images)
            has_text = bool((page.extract_text() or "").strip())
            if has_images and not has_text:
                image_table_pages.append(index)

    return {
        "path": path,
        "page_count": doc.page_count,
        "headers": unique_headers[:20],
        "paragraphs": {
            "count": len(paragraphs),
            "avg_words": round(avg_words, 1),
            "sample": sample_paragraphs,
        },
        "tables": {
            "text_tables": text_tables,
            "image_table_pages": image_table_pages,
        },
        "layout": extract_layout(path),
    }


def extract_layout(path):
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for layout extraction.")
    doc = fitz.open(path)
    layout = []
    plumber_doc = None
    if pdfplumber is not None:
        plumber_doc = pdfplumber.open(path)
    try:
        for page_num, page in enumerate(doc):
            page_layout = {"page": page_num + 1, "elements": []}
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                bbox = block.get("bbox")
                btype = block.get("type")
                if btype == 0:
                    parts = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            if text:
                                parts.append(text)
                    block_text = "".join(parts).strip()
                    page_layout["elements"].append(
                        {
                            "type": "text",
                            "bbox": bbox,
                            "text": block_text,
                        }
                    )
                elif btype == 1:
                    page_layout["elements"].append(
                        {
                            "type": "image",
                            "bbox": bbox,
                        }
                    )
            if plumber_doc is not None:
                # pdfplumber uses a bottom-left origin for coordinates; convert to fitz's top-left
                pdf_page = plumber_doc.pages[page_num]
                # height of the page in fitz coordinate space
                page_h = page.rect.height
                for img in pdf_page.images:
                    # convert bbox to fitz coordinate space
                    pb = [img["x0"], img["y0"], img["x1"], img["y1"]]
                    page_layout["elements"].append(
                        {
                            "type": "image",
                            # plumber bbox -> fitz coordinates
                            "bbox": _plumber_bbox_to_fitz(pb, page_h),
                            # mark source to aid deduplication/debugging
                            "source": "pdfplumber",
                        }
                    )
                for table in pdf_page.find_tables() or []:
                    page_layout["elements"].append(
                        {
                            "type": "table",
                            # convert table bbox to fitz coordinate space
                            "bbox": _plumber_bbox_to_fitz(list(table.bbox), page_h),
                            "cells": table.extract(),
                            "source": "pdfplumber",
                        }
                    )
            page_layout["elements"] = _postprocess_page_elements(
                page_layout["elements"], page, page_num + 1
            )
            layout.append(page_layout)
    finally:
        doc.close()
        if plumber_doc is not None:
            plumber_doc.close()
    return layout


def _postprocess_page_elements(elements, page, page_number):
    if not elements:
        return elements

    tables = [el for el in elements if el.get("type") == "table"]
    images = [el for el in elements if el.get("type") == "image"]
    texts = [el for el in elements if el.get("type") == "text"]

    page_width = page.rect.width
    page_height = page.rect.height

    # Remove tables that are likely full-page cover artifacts or fragments,
    # then drop those that are too small or sparse to be real tables.
    tables = _filter_false_tables(tables, page_width, page_height)
    tables = _filter_bad_tables(tables)
    tables = _merge_aligned_tables(tables, tolerance=10.0)

    # Remove tiny decorative images and deduplicate overlapping figures.
    images = _filter_small_images(images, page_width, page_height)
    images = _dedupe_images(images, iou_thresh=0.90)

    merged = texts + tables + images
    merged.sort(key=_element_sort_key)
    return merged


def _merge_aligned_tables(tables, *, tolerance=10.0):
    if not tables:
        return tables
    ordered = sorted(tables, key=lambda el: (el["bbox"][0], el["bbox"][1]))
    groups = []
    for table in ordered:
        bbox = table.get("bbox")
        if not bbox:
            groups.append([table])
            continue
        x0, _, x1, _ = bbox
        matched = None
        for group in groups:
            ref_bbox = group[0].get("bbox") or bbox
            if (
                abs(ref_bbox[0] - x0) <= tolerance
                and abs(ref_bbox[2] - x1) <= tolerance
            ):
                matched = group
                break
        if matched is None:
            groups.append([table])
        else:
            matched.append(table)
    merged_tables = []
    for group in groups:
        group_sorted = sorted(group, key=lambda el: (el["bbox"][1], el["bbox"][0]))
        merged_tables.append(_merge_table_group(group_sorted))
    return merged_tables


def _merge_table_group(group):
    if not group:
        return {"type": "table", "bbox": [0, 0, 0, 0], "cells": []}
    bbox = [group[0]["bbox"][0], group[0]["bbox"][1], group[0]["bbox"][2], group[0]["bbox"][3]]
    cells = []
    for table in group:
        tbbox = table.get("bbox") or [0, 0, 0, 0]
        bbox = [
            min(bbox[0], tbbox[0]),
            min(bbox[1], tbbox[1]),
            max(bbox[2], tbbox[2]),
            max(bbox[3], tbbox[3]),
        ]
        cells.extend(table.get("cells") or [])
    return {"type": "table", "bbox": bbox, "cells": cells}


def _filter_false_tables(tables, page_width, page_height, *, size_ratio=0.9):
    if not tables:
        return tables
    filtered = []
    for table in tables:
        bbox = table.get("bbox")
        if not bbox or page_width <= 0 or page_height <= 0:
            filtered.append(table)
            continue
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        width_ratio = width / page_width
        height_ratio = height / page_height
        cell_count = _table_cell_count(table.get("cells"))
        if width_ratio >= size_ratio and height_ratio >= size_ratio and cell_count <= 1:
            continue
        filtered.append(table)
    return filtered


def _dedupe_images(images, *, iou_thresh=0.90, epsilon=None):
    """Deduplicate images based on IoU overlap rather than coordinate closeness.

    Images often appear multiple times in a PDF extraction, either from
    pdfplumber and PyMuPDF both reporting the same figure or because a
    vector-based graphic is split into several segments. This function
    eliminates near-duplicate images by computing the intersection over
    union (IoU) between their bounding boxes and dropping those with
    significant overlap.

    Args:
        images: List of image element dicts with a ``bbox`` key.
        iou_thresh: IoU threshold above which an image is considered a
            duplicate of one already kept. Defaults to 0.90 (90% overlap).
        epsilon: Deprecated. Previously used for simple coordinate
            closeness; retained for backward compatibility but ignored.

    Returns:
        A list of image elements with duplicates removed.
    """
    if not images:
        return images
    # Sort by area descending so that larger images are retained over
    # smaller overlapping ones. Missing bboxes or zero-area boxes are
    # considered smallest.
    def _area(img):
        bbox = img.get("bbox") or [0, 0, 0, 0]
        x0, y0, x1, y1 = bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    ordered = sorted(images, key=_area, reverse=True)
    kept = []
    for img in ordered:
        bbox = img.get("bbox")
        if not bbox:
            kept.append(img)
            continue
        duplicate = False
        for existing in kept:
            eb = existing.get("bbox")
            if not eb:
                continue
            if _bbox_iou(bbox, eb) >= iou_thresh:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(img)
    return kept


def _bbox_close(bbox_a, bbox_b, epsilon):
    if not bbox_a or not bbox_b:
        return False
    return all(
        abs(coord_a - coord_b) <= epsilon
        for coord_a, coord_b in zip(bbox_a, bbox_b)
    )


def _element_sort_key(element):
    bbox = element.get("bbox")
    if not bbox:
        return (float("inf"), float("inf"))
    return (bbox[1], bbox[0])


def _table_cell_count(cells):
    if not cells:
        return 0
    count = 0
    for row in cells:
        if not row:
            continue
        count += len(row)
    return count

# ---------------------------------------------------------------------------
# Additional helpers for improved layout post-processing
#
# The following functions normalize coordinate systems between pdfplumber and
# PyMuPDF, filter out spurious tables and tiny decorative images, and
# deduplicate overlapping image regions using an intersection-over-union (IoU)
# threshold. These helpers make the extracted layout more faithful to the
# document's semantic structure and reduce noise in downstream RAG pipelines.

def _plumber_bbox_to_fitz(bbox, page_height):
    """Convert a pdfplumber bbox (bottom-left origin) to fitz's top-left origin.

    pdfplumber returns bounding boxes in the form [x0, y0, x1, y1] where y
    values increase upwards from the bottom of the page. PyMuPDF (fitz) uses
    a top-left origin where y values increase downwards. To reconcile these,
    we reflect the y coordinates relative to the page height.

    Args:
        bbox: List or tuple of four floats [x0, y0, x1, y1] from pdfplumber.
        page_height: Height of the page in points in the fitz coordinate space.

    Returns:
        A list [x0, y_top, x1, y_bottom] in fitz coordinate space.
    """
    if not bbox or len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = bbox
    return [x0, page_height - y1, x1, page_height - y0]


def _bbox_iou(bbox_a, bbox_b):
    """Compute the intersection over union (IoU) between two bounding boxes.

    Args:
        bbox_a: [x0, y0, x1, y1] for the first box.
        bbox_b: [x0, y0, x1, y1] for the second box.

    Returns:
        A float between 0 and 1 representing the IoU. Values near 1 indicate
        strong overlap; values near 0 indicate little or no overlap.
    """
    if not bbox_a or not bbox_b:
        return 0.0
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    # intersection
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _filter_small_images(images, page_width, page_height, *, min_area_ratio=0.002):
    """Remove images whose bounding box area is too small relative to the page.

    Many PDFs contain tiny decorative elements (icons, ticks, bullets) stored
    as images. These should not be treated as separate figures. This helper
    drops any image whose area is below a percentage of the page area.

    Args:
        images: List of image element dicts with a ``bbox`` key.
        page_width: Width of the page in points.
        page_height: Height of the page in points.
        min_area_ratio: Minimum fraction of the page area an image must cover
            to be kept. Defaults to 0.2% (0.002).

    Returns:
        A filtered list of image elements.
    """
    if not images:
        return images
    page_area = page_width * page_height
    if page_area <= 0:
        return images
    filtered = []
    for el in images:
        bbox = el.get("bbox")
        if not bbox:
            filtered.append(el)
            continue
        x0, y0, x1, y1 = bbox
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if page_area > 0 and (area / page_area) < min_area_ratio:
            # Discard very small images
            continue
        filtered.append(el)
    return filtered


def _table_nonempty_cell_ratio(cells):
    """Compute the ratio of non-empty cells to total cells in a table."""
    if not cells:
        return 0.0
    flat = []
    for row in cells or []:
        if not row:
            continue
        for cell in row:
            flat.append(cell)
    if not flat:
        return 0.0
    nonempty = sum(1 for c in flat if str(c).strip())
    return nonempty / len(flat)


def _filter_bad_tables(tables, *, min_cells=6, min_nonempty_ratio=0.3):
    """Filter out table elements that are likely false positives or fragments.

    Args:
        tables: List of table element dicts with ``cells`` and ``bbox``.
        min_cells: Minimum total cell count required for a table to be kept.
        min_nonempty_ratio: Minimum ratio of non-empty cells to total cells.

    Returns:
        A list of filtered table elements.
    """
    if not tables:
        return tables
    filtered = []
    for table in tables:
        cells = table.get("cells") or []
        if _table_cell_count(cells) < min_cells:
            continue
        if _table_nonempty_cell_ratio(cells) < min_nonempty_ratio:
            continue
        filtered.append(table)
    return filtered




def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Notes de recherche and analyze PDFs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape notes de recherche.")
    scrape_parser.add_argument("--url", default=DEFAULT_URL)
    scrape_parser.add_argument(
        "--min-date", default="2026-01-01", help="Filter date in YYYY-MM-DD."
    )
    scrape_parser.add_argument("--use-selenium", action="store_true")
    scrape_parser.add_argument("--timeout", type=int, default=30)
    scrape_parser.add_argument("--out", help="Write JSON results to a file.")
    scrape_parser.add_argument(
        "--download-dir", help="If set, download PDF files to this directory."
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a PDF.")
    analyze_parser.add_argument("--pdf", required=True, help="Path to a PDF file.")
    analyze_parser.add_argument("--out", help="Write analysis to a file.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "scrape":
        min_date = dt.datetime.strptime(args.min_date, "%Y-%m-%d").date()
        items = scrape_notes(
            args.url, min_date, use_selenium=args.use_selenium, timeout=args.timeout
        )
        if args.download_dir:
            for item in items:
                try:
                    if item.get("pdf_link"):
                        item["downloaded_path"] = download_pdf(
                            item["pdf_link"], args.download_dir
                        )
                except Exception:
                    item["downloaded_path"] = None
        payload = json.dumps(items, indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            print(payload)
        return 0

    if args.command == "analyze":
        result = analyze_pdf(args.pdf)
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            print(payload)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
