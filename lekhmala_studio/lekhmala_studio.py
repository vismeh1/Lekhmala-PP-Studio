"""
Lekhmala Photo Studio — Reflex Port
Converted from Streamlit (pro_studio.py) by Claude
"""

import os
import sys
import io
import base64
import urllib.request

import reflex as rx
import cv2
import numpy as np
from PIL import Image, ImageDraw
from rembg import remove

# ── Cloud / graphics safety net ──────────────────────────────────────────────
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        from torchvision.transforms import functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F
    except Exception:
        pass

from gfpgan import GFPGANer  # noqa: E402 (must come after sys.modules patch)

# ── Constants ─────────────────────────────────────────────────────────────────
PX_PER_MM: float = 23.622  # 600 DPI

CANVAS_SIZES: dict[str, tuple[float, float]] = {
    "A4 Sheet": (210, 297),
    "4x6 Inch (Photo Paper)": (101.6, 152.4),
    "A3 Sheet": (297, 420),
}

PHOTO_TYPES: dict[str, tuple[float, float] | None] = {
    "Standard Passport (35x45mm)": (35, 45),
    "US Visa (2x2 inch)": (50.8, 50.8),
    "Stamp Size (20x25mm)": (20, 25),
    "Custom Size": None,
}

# ── AI model singleton (lazy-loaded once per process) ─────────────────────────
_enhancer: GFPGANer | None = None


def _get_enhancer() -> GFPGANer:
    global _enhancer
    if _enhancer is None:
        model_path = "GFPGANv1.4.pth"
        if not os.path.exists(model_path):
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            urllib.request.urlretrieve(url, model_path)
        _enhancer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
        )
    return _enhancer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _build_single(
    subject: Image.Image,
    target_dim: tuple[float, float],
    zoom: float,
    move_x: int,
    move_y: int,
    show_border: bool,
) -> Image.Image:
    """Crop / resize / optionally border one passport photo."""
    sw_px = int(target_dim[0] * PX_PER_MM)
    sh_px = int(target_dim[1] * PX_PER_MM)
    img_w, img_h = subject.size
    crop_w = img_w / zoom
    crop_h = (crop_w * sh_px) / sw_px
    left = (img_w / 2 + move_x) - crop_w / 2
    top = (img_h / 2 + move_y) - crop_h / 2
    photo = subject.crop((left, top, left + crop_w, top + crop_h))
    photo = photo.resize((sw_px, sh_px), Image.Resampling.LANCZOS)
    if show_border:
        draw = ImageDraw.Draw(photo)
        draw.rectangle([0, 0, sw_px - 1, sh_px - 1], outline="black", width=4)
    return photo


def _build_sheet(
    single: Image.Image,
    paper_choice: str,
    orientation: str,
    num_copies: int,
    gap_h: float,
    gap_v: float,
    margin: float,
) -> tuple[Image.Image, int]:
    """Tile `single` onto a print canvas; return (canvas, placed_count)."""
    sw_px, sh_px = single.size
    p_w_mm, p_h_mm = CANVAS_SIZES[paper_choice]
    if orientation == "Landscape":
        cw_px, ch_px = int(p_h_mm * PX_PER_MM), int(p_w_mm * PX_PER_MM)
    else:
        cw_px, ch_px = int(p_w_mm * PX_PER_MM), int(p_h_mm * PX_PER_MM)

    canvas = Image.new("RGB", (cw_px, ch_px), "white")
    m_px = int(margin * PX_PER_MM)
    gh_px = int(gap_h * PX_PER_MM)
    gv_px = int(gap_v * PX_PER_MM)

    curr_x, curr_y, placed = m_px, m_px, 0
    for _ in range(num_copies):
        if curr_x + sw_px > cw_px - m_px:
            curr_x = m_px
            curr_y += sh_px + gv_px
        if curr_y + sh_px > ch_px - m_px:
            break
        canvas.paste(single, (int(curr_x), int(curr_y)))
        curr_x += sw_px + gh_px
        placed += 1

    return canvas, placed


# ── State ─────────────────────────────────────────────────────────────────────

class State(rx.State):

    # ── Step-1 config ──
    paper_choice: str = "A4 Sheet"
    orientation: str = "Landscape"
    photo_choice: str = "Standard Passport (35x45mm)"
    custom_w: float = 35.0
    custom_h: float = 45.0

    # ── Step-2 adjustments ──
    zoom: float = 1.25
    move_x: int = 0
    move_y: int = 0
    show_border: bool = True

    # ── Step-3 export ──
    num_copies: int = 12
    gap_h: float = 2.0
    gap_v: float = 2.0
    margin: float = 5.0
    pdf_mode: str = "RGB"

    # ── Processing flags ──
    is_processing: bool = False
    status_msg: str = ""
    upload_error: str = ""
    last_uploaded: str = ""

    # ── Image store (base64 PNG/JPEG strings) ──
    processed_b64: str = ""   # white-bg, full-res subject
    preview_b64: str = ""     # single cropped photo
    sheet_b64: str = ""       # print sheet
    dl_jpg_b64: str = ""
    dl_png_b64: str = ""
    dl_pdf_b64: str = ""

    # ── Computed vars ─────────────────────────────────────────────────────────

    @rx.var
    def target_dim(self) -> tuple[float, float]:
        if self.photo_choice == "Custom Size":
            return (self.custom_w, self.custom_h)
        return PHOTO_TYPES[self.photo_choice]  # type: ignore[return-value]

    @rx.var
    def has_processed(self) -> bool:
        return self.processed_b64 != ""

    @rx.var
    def has_sheet(self) -> bool:
        return self.sheet_b64 != ""

    @rx.var
    def preview_data_url(self) -> str:
        return f"data:image/png;base64,{self.preview_b64}" if self.preview_b64 else ""

    @rx.var
    def sheet_data_url(self) -> str:
        return f"data:image/png;base64,{self.sheet_b64}" if self.sheet_b64 else ""

    # ── Upload & AI processing ─────────────────────────────────────────────────

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return

        file = files[0]
        raw = await file.read()

        if file.filename == self.last_uploaded and self.processed_b64:
            return  # already processed this file

        self.is_processing = True
        self.upload_error = ""
        self.status_msg = "Initializing Face Restoration…"
        yield

        try:
            file_bytes = np.frombuffer(raw, np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            self.status_msg = "Enhancing Biometric Quality…"
            yield

            try:
                enhancer = _get_enhancer()
                _, _, enhanced = enhancer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True,
                )
                pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            except Exception:
                # Graceful fallback — skip GFPGAN, use raw decode
                pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            self.status_msg = "Generating Studio White Background…"
            yield

            no_bg = remove(pil_img)
            canvas = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
            canvas.paste(no_bg, (0, 0), mask=no_bg)
            processed = canvas.convert("RGB")

            self.processed_b64 = _pil_to_b64(processed)
            self.last_uploaded = file.filename
            self.status_msg = "✅ AI Processing Complete!"

            # Immediately render preview with current slider values
            self._refresh_preview(processed)

        except Exception as exc:
            self.upload_error = f"Processing failed: {exc}"

        finally:
            self.is_processing = False

    # ── Preview refresh ────────────────────────────────────────────────────────

    def _refresh_preview(self, subject: Image.Image | None = None):
        if subject is None:
            if not self.processed_b64:
                return
            subject = _b64_to_pil(self.processed_b64)

        single = _build_single(
            subject,
            self.target_dim,
            self.zoom,
            self.move_x,
            self.move_y,
            self.show_border,
        )
        self.preview_b64 = _pil_to_b64(single)

    def _load_and_refresh(self):
        """Load subject from stored b64, then refresh preview."""
        self._refresh_preview()

    # ── Slider / control event handlers ───────────────────────────────────────

    def set_zoom(self, val: float):
        self.zoom = val
        self._load_and_refresh()

    def set_move_x(self, val: int):
        self.move_x = val
        self._load_and_refresh()

    def set_move_y(self, val: int):
        self.move_y = val
        self._load_and_refresh()

    def set_show_border(self, val: bool):
        self.show_border = val
        self._load_and_refresh()

    def set_photo_choice(self, val: str):
        self.photo_choice = val
        self._load_and_refresh()

    def set_paper_choice(self, val: str):
        self.paper_choice = val

    def set_orientation(self, val: str):
        self.orientation = val

    def set_custom_w(self, val: float):
        self.custom_w = val
        self._load_and_refresh()

    def set_custom_h(self, val: float):
        self.custom_h = val
        self._load_and_refresh()

    def set_gap_h(self, val: float):
        self.gap_h = val

    def set_gap_v(self, val: float):
        self.gap_v = val

    def set_margin(self, val: float):
        self.margin = val

    def set_num_copies(self, val: str):
        try:
            self.num_copies = max(1, min(300, int(val)))
        except ValueError:
            pass

    def set_pdf_mode(self, val: str):
        self.pdf_mode = val

    # ── Sheet generation & downloads ──────────────────────────────────────────

    def generate_sheet(self):
        if not self.preview_b64:
            return
        single = _b64_to_pil(self.preview_b64)
        canvas, _ = _build_sheet(
            single,
            self.paper_choice,
            self.orientation,
            self.num_copies,
            self.gap_h,
            self.gap_v,
            self.margin,
        )
        self.sheet_b64 = _pil_to_b64(canvas)

        # JPG
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=100)
        self.dl_jpg_b64 = base64.b64encode(buf.getvalue()).decode()

        # PNG
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        self.dl_png_b64 = base64.b64encode(buf.getvalue()).decode()

        # PDF
        buf = io.BytesIO()
        pdf_img = canvas.copy()
        if self.pdf_mode == "CMYK":
            pdf_img = pdf_img.convert("CMYK")
        pdf_img.save(buf, format="PDF", resolution=600.0)
        self.dl_pdf_b64 = base64.b64encode(buf.getvalue()).decode()

    def download_jpg(self):
        return rx.download(
            data=base64.b64decode(self.dl_jpg_b64),
            filename="Lekhmala_Studio.jpg",
        )

    def download_png(self):
        return rx.download(
            data=base64.b64decode(self.dl_png_b64),
            filename="Lekhmala_Studio.png",
        )

    def download_pdf(self):
        return rx.download(
            data=base64.b64decode(self.dl_pdf_b64),
            filename="Lekhmala_Studio.pdf",
        )


# ── UI helpers ────────────────────────────────────────────────────────────────

def _label(text: str) -> rx.Component:
    return rx.text(text, font_size="0.85em", color="#444", margin_bottom="2px")


def _card(*children, **props) -> rx.Component:
    return rx.box(
        *children,
        padding="1em",
        border="1px solid #ddd",
        border_radius="8px",
        background="white",
        **props,
    )


# ── Tab 1: Upload ─────────────────────────────────────────────────────────────

def upload_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            _card(
                rx.text("📄 Paper Setup", font_weight="bold", margin_bottom="0.5em"),
                _label("Paper Size"),
                rx.select(
                    list(CANVAS_SIZES.keys()),
                    value=State.paper_choice,
                    on_change=State.set_paper_choice,
                    width="100%",
                ),
                rx.box(height="0.5em"),
                _label("Orientation"),
                rx.radio_group(
                    ["Portrait", "Landscape"],
                    value=State.orientation,
                    on_change=State.set_orientation,
                    direction="row",
                    spacing="4",
                ),
                flex="1",
            ),
            _card(
                rx.text("🖼️ Photo Standard", font_weight="bold", margin_bottom="0.5em"),
                _label("Photo Type"),
                rx.select(
                    list(PHOTO_TYPES.keys()),
                    value=State.photo_choice,
                    on_change=State.set_photo_choice,
                    width="100%",
                ),
                rx.cond(
                    State.photo_choice == "Custom Size",
                    rx.hstack(
                        rx.vstack(
                            _label("Width (mm)"),
                            rx.input(
                                value=State.custom_w.to_string(),
                                on_change=State.set_custom_w,
                                type="number",
                                width="100%",
                            ),
                        ),
                        rx.vstack(
                            _label("Height (mm)"),
                            rx.input(
                                value=State.custom_h.to_string(),
                                on_change=State.set_custom_h,
                                type="number",
                                width="100%",
                            ),
                        ),
                        width="100%",
                        spacing="3",
                    ),
                ),
                flex="1",
            ),
            width="100%",
            spacing="4",
            align_items="stretch",
        ),
        rx.upload(
            rx.vstack(
                rx.icon("upload", size=32, color="#007BFF"),
                rx.text(
                    "📤 Upload Portrait Photo",
                    font_weight="bold",
                    color="#007BFF",
                ),
                rx.text(
                    "Drag & drop or click to browse  •  JPG / PNG",
                    font_size="0.85em",
                    color="#888",
                ),
                align="center",
                spacing="2",
            ),
            id="photo_upload",
            accept={
                "image/jpeg": [".jpg", ".jpeg"],
                "image/png": [".png"],
            },
            on_drop=State.handle_upload(
                rx.upload_files(upload_id="photo_upload")
            ),
            border="2px dashed #007BFF",
            border_radius="10px",
            padding="2.5em",
            width="100%",
            text_align="center",
            cursor="pointer",
            _hover={"background": "#e8f0fe"},
            transition="background 0.2s",
        ),
        # Processing feedback
        rx.cond(
            State.is_processing,
            rx.hstack(
                rx.spinner(size="2"),
                rx.text(State.status_msg, color="#007BFF"),
                align="center",
                spacing="3",
            ),
            rx.cond(
                State.status_msg != "",
                rx.text(State.status_msg, color="green", font_size="0.9em"),
            ),
        ),
        rx.cond(
            State.upload_error != "",
            rx.callout(
                State.upload_error,
                icon="triangle_alert",
                color_scheme="red",
            ),
        ),
        width="100%",
        spacing="4",
    )


# ── Tab 2: Adjust ─────────────────────────────────────────────────────────────

def adjust_tab() -> rx.Component:
    return rx.cond(
        State.has_processed,
        rx.hstack(
            _card(
                rx.text(
                    "✂️ Frame & Biometric Alignment",
                    font_weight="bold",
                    font_size="1.05em",
                    margin_bottom="0.5em",
                ),
                rx.callout(
                    "Nepal Standard: Head should fill 70–80% of frame height.",
                    icon="info",
                    color_scheme="blue",
                    margin_bottom="0.75em",
                ),
                _label(f"Zoom / Face Size  ({State.zoom})"),
                rx.slider(
                    min=0.5,
                    max=4.0,
                    step=0.05,
                    default_value=[1.25],
                    on_value_commit=State.set_zoom,
                    width="100%",
                ),
                rx.box(height="0.4em"),
                _label(f"Vertical Position  ({State.move_y})"),
                rx.slider(
                    min=-1500,
                    max=1500,
                    step=10,
                    default_value=[0],
                    on_value_commit=State.set_move_y,
                    width="100%",
                ),
                rx.box(height="0.4em"),
                _label(f"Horizontal Position  ({State.move_x})"),
                rx.slider(
                    min=-1500,
                    max=1500,
                    step=10,
                    default_value=[0],
                    on_value_commit=State.set_move_x,
                    width="100%",
                ),
                rx.box(height="0.75em"),
                rx.checkbox(
                    "Add Photo Border",
                    checked=State.show_border,
                    on_change=State.set_show_border,
                ),
                flex="1",
                min_width="280px",
            ),
            _card(
                rx.cond(
                    State.preview_data_url != "",
                    rx.vstack(
                        rx.image(
                            src=State.preview_data_url,
                            width="220px",
                            border_radius="4px",
                            box_shadow="0 2px 8px rgba(0,0,0,0.15)",
                        ),
                        rx.text(
                            "Live Biometric Preview",
                            font_size="0.8em",
                            color="#888",
                        ),
                        align="center",
                        spacing="2",
                    ),
                    rx.text("Preview loading…", color="#aaa"),
                ),
                flex="1",
                display="flex",
                align_items="center",
                justify_content="center",
            ),
            width="100%",
            spacing="4",
            align_items="flex-start",
        ),
        rx.callout(
            "👋 Please upload a photo in Step 1 first.",
            icon="upload",
            color_scheme="orange",
        ),
    )


# ── Tab 3: Export ─────────────────────────────────────────────────────────────

def export_tab() -> rx.Component:
    return rx.cond(
        State.has_processed,
        rx.vstack(
            rx.hstack(
                _card(
                    rx.text("📋 Layout Settings", font_weight="bold", margin_bottom="0.5em"),
                    _label("Number of Copies"),
                    rx.input(
                        value=State.num_copies.to_string(),
                        on_change=State.set_num_copies,
                        type="number",
                        width="100%",
                    ),
                    rx.box(height="0.4em"),
                    _label(f"Horizontal Gap  ({State.gap_h} mm)"),
                    rx.slider(
                        min=0.0,
                        max=20.0,
                        step=0.5,
                        default_value=[2.0],
                        on_value_commit=State.set_gap_h,
                        width="100%",
                    ),
                    rx.box(height="0.4em"),
                    _label(f"Vertical Gap  ({State.gap_v} mm)"),
                    rx.slider(
                        min=0.0,
                        max=20.0,
                        step=0.5,
                        default_value=[2.0],
                        on_value_commit=State.set_gap_v,
                        width="100%",
                    ),
                    rx.box(height="0.4em"),
                    _label(f"Page Margin  ({State.margin} mm)"),
                    rx.slider(
                        min=2.0,
                        max=30.0,
                        step=0.5,
                        default_value=[5.0],
                        on_value_commit=State.set_margin,
                        width="100%",
                    ),
                    rx.box(height="0.4em"),
                    _label("Export Color Profile"),
                    rx.select(
                        ["RGB", "CMYK"],
                        value=State.pdf_mode,
                        on_change=State.set_pdf_mode,
                        width="100%",
                    ),
                    rx.box(height="0.75em"),
                    rx.button(
                        "🖨️ Generate Print Sheet",
                        on_click=State.generate_sheet,
                        background_color="#007BFF",
                        color="white",
                        width="100%",
                        border_radius="6px",
                        padding_y="0.6em",
                        font_weight="bold",
                        _hover={"background_color": "#0056b3"},
                    ),
                    flex="1",
                    min_width="260px",
                ),
                _card(
                    rx.cond(
                        State.has_sheet,
                        rx.image(
                            src=State.sheet_data_url,
                            width="100%",
                            border_radius="4px",
                        ),
                        rx.vstack(
                            rx.icon("image", size=48, color="#ccc"),
                            rx.text(
                                "Configure layout and click Generate",
                                color="#aaa",
                                font_size="0.9em",
                            ),
                            align="center",
                            spacing="2",
                        ),
                    ),
                    flex="2",
                    display="flex",
                    align_items="center",
                    justify_content="center",
                    min_height="300px",
                ),
                width="100%",
                spacing="4",
                align_items="flex-start",
            ),
            rx.cond(
                State.has_sheet,
                rx.hstack(
                    rx.button(
                        "📥 Ultra-HD JPG",
                        on_click=State.download_jpg,
                        background_color="#28a745",
                        color="white",
                        border_radius="6px",
                        font_weight="bold",
                        _hover={"background_color": "#1e7e34"},
                    ),
                    rx.button(
                        "📥 Lossless PNG",
                        on_click=State.download_png,
                        background_color="#28a745",
                        color="white",
                        border_radius="6px",
                        font_weight="bold",
                        _hover={"background_color": "#1e7e34"},
                    ),
                    rx.button(
                        rx.text(f"📥 {State.pdf_mode} PDF"),
                        on_click=State.download_pdf,
                        background_color="#28a745",
                        color="white",
                        border_radius="6px",
                        font_weight="bold",
                        _hover={"background_color": "#1e7e34"},
                    ),
                    spacing="4",
                    flex_wrap="wrap",
                ),
            ),
            width="100%",
            spacing="4",
        ),
        rx.callout(
            "👋 Please upload a photo in Step 1 first.",
            icon="upload",
            color_scheme="orange",
        ),
    )


# ── Page ──────────────────────────────────────────────────────────────────────

def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Header
            rx.vstack(
                rx.heading(
                    "Lekhmala Photo Studio",
                    size="8",
                    color="#1E3A8A",
                ),
                rx.text(
                    "Professional PP Photo Solution  |  Developed by Bishal Mehta",
                    color="#555",
                    text_align="center",
                ),
                align="center",
                margin_bottom="1em",
            ),
            # Main tabs
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger(
                        "🚀 Step 1: Upload",
                        value="upload",
                    ),
                    rx.tabs.trigger(
                        "🎨 Step 2: Adjust",
                        value="adjust",
                        disabled=~State.has_processed,
                    ),
                    rx.tabs.trigger(
                        "📥 Step 3: Export",
                        value="export",
                        disabled=~State.has_processed,
                    ),
                    margin_bottom="1em",
                ),
                rx.tabs.content(upload_tab(), value="upload", padding_top="1em"),
                rx.tabs.content(adjust_tab(), value="adjust", padding_top="1em"),
                rx.tabs.content(export_tab(), value="export", padding_top="1em"),
                default_value="upload",
                width="100%",
            ),
            # Footer
            rx.divider(margin_y="1.5em"),
            rx.text(
                "Lekhmala Photo Studio v2.5  |  Professional PP Photo Solution "
                "Optimized for Nepal Gov Standards  |  Developed by Bishal Mehta",
                color="#aaa",
                font_size="0.78em",
                text_align="center",
            ),
            max_width="1200px",
            width="100%",
            margin="0 auto",
            padding="2em",
            spacing="0",
        ),
        background_color="#f8f9fa",
        min_height="100vh",
    )


# ── App entry point ───────────────────────────────────────────────────────────

app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        radius="medium",
        accent_color="blue",
    )
)
app.add_page(index, route="/", title="Lekhmala Photo Studio")
