"""Download freely licensed skill media and background reading for the AI coach.

    Wikimedia Commons files in skill_guide.MEDIA -> out/demo/media/  (+ manifest.json with credits)
    Wikipedia articles in skill_guide.WIKIPEDIA  -> out/skill_wiki.json (plain-text extracts)

Only public-domain / CC0 / CC BY / CC BY-SA files are kept; every file keeps its
author, licence and source link so the page can credit it.

Usage: python -m skilltwin.skill_media
"""

import html
import json
import re

import requests

from skilltwin.config import OUT_DIR
from skilltwin.skill_guide import MEDIA, WIKIPEDIA

MEDIA_DIR = OUT_DIR / "demo" / "media"
HEADERS = {"User-Agent": "SkillTwinHackathon/0.1 (educational demo; contact: sage1419a@gmail.com)"}
COMMONS = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 420
FREE = ("public domain", "cc0", "cc by", "cc-by", "pd")
WIKI_CHARS = 6000


def is_free(lic):
    lic = lic.lower()
    return any(k in lic for k in FREE) and "-nc" not in lic and "-nd" not in lic and " nc" not in lic and " nd" not in lic


def plain(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


def file_info(names):
    r = requests.get(COMMONS, headers=HEADERS, timeout=60, params={
        "action": "query", "titles": "|".join("File:" + n for n in names), "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": THUMB_WIDTH, "format": "json"}).json()
    out = {}
    for p in r["query"]["pages"].values():
        if "imageinfo" in p:
            out[p["title"][5:]] = p["imageinfo"][0]
    for n in r["query"].get("normalized", []):
        if n["to"][5:] in out:
            out[n["from"][5:]] = out[n["to"][5:]]
    return out


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def download_media():
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted({n for files in MEDIA.values() for n, _ in files})
    info = file_info(names)
    files = {}
    for name in names:
        ii = info.get(name)
        if not ii:
            print(f"  missing on Commons: {name}")
            continue
        meta = ii["extmetadata"]
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if not is_free(lic):
            print(f"  skipped (licence {lic}): {name}")
            continue
        video = ii["mime"].startswith("video/")
        # animated GIFs keep their original file (thumbnails lose the animation)
        full = video or ii["mime"] == "image/gif" and ii["size"] > 500_000
        url = ii["url"] if full else ii.get("thumburl", ii["url"])
        ext = "." + url.split("?")[0].rsplit(".", 1)[-1].lower()
        local = safe_name(name.rsplit(".", 1)[0]) + ext
        path = MEDIA_DIR / local
        if not path.exists():
            data = requests.get(url, headers=HEADERS, timeout=600).content
            path.write_bytes(data)
        files[name] = {
            "file": local, "kind": "video" if video else "image", "bytes": path.stat().st_size,
            "title": name.rsplit(".", 1)[0], "author": plain(meta.get("Artist", {}).get("value")) or "Unknown",
            "license": lic, "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "source": ii.get("descriptionurl", ""),
        }
        print(f"  {files[name]['kind']:5s} {path.stat().st_size // 1024:6d} KB  {lic:14s} {name}")
    manifest = {fam: [{**files[n], "local_only": lo} for n, lo in entries if n in files]
                for fam, entries in MEDIA.items()}
    (MEDIA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def download_wiki():
    out = {}
    for title in sorted(set(WIKIPEDIA.values())):
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=60, params={
            "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
            "titles": title, "format": "json"}).json()
        page = next(iter(r["query"]["pages"].values()))
        text = page.get("extract", "")
        text = re.split(r"\n== (See also|References|External links|Further reading|Notes) ==", text)[0]
        out[title] = {"title": page.get("title", title), "text": text[:WIKI_CHARS],
                      "url": "https://en.wikipedia.org/wiki/" + page.get("title", title).replace(" ", "_")}
        print(f"  wikipedia {out[title]['title']:28s} {len(text):6d} chars (kept {len(out[title]['text'])})")
    (OUT_DIR / "skill_wiki.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    print("Wikimedia Commons media:")
    download_media()
    print("Wikipedia background reading:")
    download_wiki()


if __name__ == "__main__":
    main()
