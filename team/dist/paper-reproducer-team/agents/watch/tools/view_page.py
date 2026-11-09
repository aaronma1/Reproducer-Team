# -*- coding: utf-8 -*-
"""view_page: render one page of a PDF (optionally a cropped panel) to a JPEG the model can open.

read_file gives a PDF's text, which keeps a plot's tick labels but not its curves. This renders the page into
``<dir of the PDF>/.pages/`` and returns the image path; open it with read_file to see it. A cropped panel is
enlarged so small tick labels stay legible. Ported from ``framework/tools.py`` ``Scope.render_page``.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, AsyncIterator

from openjiuwen.core.foundation.tool.base import Tool, ToolCard

_PDFIUM_LOCK = threading.Lock()  # pdfium is not thread-safe; parallel figure readers crashed runs without it


def render(pdf: str, page: int, crop: str = "", scale: float = 2.0) -> str:
    path = Path(pdf)
    if path.suffix.lower() != ".pdf" or path.is_symlink() or not path.is_file():
        raise ValueError("view_page only renders a regular PDF file")
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(path))
        try:
            if not 1 <= int(page) <= len(doc):
                raise ValueError(f"page {page} is out of range 1..{len(doc)}")
            img = doc[int(page) - 1].render(scale=scale).to_pil()
        finally:
            doc.close()
    if crop:
        x0, y0, x1, y1 = (float(v) for v in crop.split(","))
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("crop must be x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        w, h = img.size
        img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        if img.width < 1200:
            k = 1200 / img.width
            img = img.resize((1200, int(img.height * k)))
    out_dir = path.parent / ".pages"
    out_dir.mkdir(exist_ok=True)
    tag = hashlib.sha1(f"{path.stat().st_mtime_ns}:{page}:{crop}:{scale}".encode()).hexdigest()[:10]
    out = out_dir / f"page{int(page)}-{tag}.jpg"
    if not out.exists():
        img.convert("RGB").save(out, format="JPEG", quality=85)
    return str(out)


def _args(inputs: Any) -> dict:
    if isinstance(inputs, dict):
        return inputs
    dump = getattr(inputs, "model_dump", None)
    data = dump() if callable(dump) else None
    return data if isinstance(data, dict) else {}


class ViewPage(Tool):
    def __init__(self) -> None:
        super().__init__(ToolCard(
            id="view_page", name="view_page",
            description=("Render one page of a PDF (e.g. the run's paper.pdf) as a JPEG and return its path; open the "
                         "path with read_file to look at it. Use `crop` (x0,y0,x1,y1 as fractions of the page, 0,0 top "
                         "left) to zoom into one figure panel. read_file on a PDF gives its text only, without curves."),
            input_params={"type": "object", "additionalProperties": False, "required": ["file_path", "page"],
                          "properties": {"file_path": {"type": "string", "description": "absolute path of the PDF"},
                                         "page": {"type": "integer", "description": "1-based page number"},
                                         "crop": {"type": "string", "description": "optional x0,y0,x1,y1"}}}))

    async def invoke(self, inputs: Any, **kwargs: Any) -> dict:
        del kwargs
        a = _args(inputs)
        try:
            image = render(str(a.get("file_path") or ""), int(a.get("page") or 0), str(a.get("crop") or ""))
        except Exception as exc:  # noqa: BLE001 - the model gets the reason and can retry
            return {"error": str(exc)}
        return {"image_path": image, "next": "open image_path with read_file to see the page"}

    async def stream(self, inputs: Any, **kwargs: Any) -> AsyncIterator[dict]:
        yield await self.invoke(inputs, **kwargs)
