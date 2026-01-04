import sys
import subprocess
import tempfile
import threading
import os
from pathlib import Path
from PIL import Image
from PySide6.QtCore import Qt, QObject, Signal, QThread, QSettings
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QMessageBox, QStackedWidget, QProgressBar
)

def img_open_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
    if im.mode == "RGBA":
        im = im.convert("RGB")
    return im

def img_open_luma_8bit(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return im

def img_open_rgba_if_any(path: str):
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
    has_alpha = False
    if "A" in im.getbands():
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        a = im.getchannel("A")
        mn, mx = a.getextrema()
        has_alpha = mn < 255
    if not has_alpha:
        if im.mode != "RGB":
            im = im.convert("RGB")
    return im, has_alpha

def resize_to(im: Image.Image, size) -> Image.Image:
    if im.size == size:
        return im
    return im.resize(size, Image.Resampling.LANCZOS)

def build_levels_lut(black_in: int, gamma: float, white_in: int):
    b = max(0, min(255, int(black_in)))
    w = max(0, min(255, int(white_in)))
    if b >= 255:
        b = 254
    if w <= b:
        w = b + 1
    g = float(gamma) if gamma and gamma > 0 else 1.0
    denom = float(w - b)
    lut = []
    for v in range(256):
        x = (v - b) / denom
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        y = pow(x, g)
        out = int(y * 255.0 + 0.5)
        if out < 0:
            out = 0
        elif out > 255:
            out = 255
        lut.append(out)
    return lut

def apply_levels_l_8bit(im_l: Image.Image, black_in: int, gamma: float, white_in: int) -> Image.Image:
    if im_l.mode != "L":
        im_l = im_l.convert("L")
    lut = build_levels_lut(black_in, gamma, white_in)
    out = im_l.point(lut)
    if out.mode != "L":
        out = out.convert("L")
    return out

def stats_l(im_l: Image.Image):
    if im_l.mode != "L":
        im_l = im_l.convert("L")
    hist = im_l.histogram()
    total = sum(hist) if hist else 1
    mn, mx = im_l.getextrema()
    mean = 0.0
    for i, c in enumerate(hist):
        if c:
            mean += i * c
    mean /= float(total)

    def pct(p):
        target = p * total
        s = 0
        for i, c in enumerate(hist):
            s += c
            if s >= target:
                return i
        return 255

    unique = sum(1 for c in hist if c)
    c0 = hist[0] if hist else 0
    c255 = hist[255] if hist else 0
    return {
        "min": int(mn),
        "max": int(mx),
        "mean": mean,
        "p05": int(pct(0.05)),
        "p50": int(pct(0.50)),
        "p95": int(pct(0.95)),
        "unique": int(unique),
        "c0": int(c0),
        "c255": int(c255),
        "total": int(total)
    }

def hist_distance(a: Image.Image, b: Image.Image):
    if a.mode != "L":
        a = a.convert("L")
    if b.mode != "L":
        b = b.convert("L")
    ha = a.histogram()
    hb = b.histogram()
    tot = sum(ha) if ha else 1
    d = 0
    for i in range(256):
        d += abs((ha[i] if ha else 0) - (hb[i] if hb else 0))
    return d / float(tot)

def height_to_normal(height_l: Image.Image, strength: float) -> Image.Image:
    h = height_l.convert("L")
    w, hgt = h.size
    px = h.load()

    def getv(x, y):
        if x < 0:
            x = 0
        elif x >= w:
            x = w - 1
        if y < 0:
            y = 0
        elif y >= hgt:
            y = hgt - 1
        return px[x, y] / 255.0

    out = Image.new("RGB", (w, hgt))
    out_px = out.load()
    s = float(strength)

    for y in range(hgt):
        for x in range(w):
            hl = getv(x - 1, y)
            hr = getv(x + 1, y)
            hu = getv(x, y - 1)
            hd = getv(x, y + 1)

            dx = (hl - hr) * s
            dy = (hu - hd) * s
            dz = 1.0

            inv_len = 1.0 / ((dx * dx + dy * dy + dz * dz) ** 0.5)
            nx = dx * inv_len
            ny = dy * inv_len
            nz = dz * inv_len

            r = int((nx * 0.5 + 0.5) * 255.0 + 0.5)
            g = int((ny * 0.5 + 0.5) * 255.0 + 0.5)
            b = int((nz * 0.5 + 0.5) * 255.0 + 0.5)

            if r < 0: r = 0
            if r > 255: r = 255
            if g < 0: g = 0
            if g > 255: g = 255
            if b < 0: b = 0
            if b > 255: b = 255

            out_px[x, y] = (r, g, b)
    return out

def norm_invert_green(im_rgb: Image.Image) -> Image.Image:
    r, g, b = im_rgb.split()
    g = Image.eval(g, lambda v: 255 - v)
    return Image.merge("RGB", (r, g, b))

def is_vtf(p: str):
    return p.lower().endswith(".vtf")

def maretf_extract_to_png(maretf_path: str, in_vtf: Path, out_png: Path, log_fn):
    exe = Path(maretf_path)
    if not exe.exists():
        raise FileNotFoundError(str(exe))
    args = [str(exe), "extract", str(in_vtf), "-o", str(out_png), "--extract-format", "png", "--no-pretty-formatting", "--verbose"]
    log_fn("MareTF: " + " ".join(args))
    cp = subprocess.run(args, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError((cp.stdout or "") + "\n" + (cp.stderr or ""))
    if not out_png.exists():
        raise RuntimeError("MareTF extract: PNG non généré.")

def maretf_create_vtf(maretf_path: str, in_img: Path, out_vtf: Path, fmt: str, srgb: bool, flags, log_fn):
    exe = Path(maretf_path)
    if not exe.exists():
        raise FileNotFoundError(str(exe))
    args = [str(exe), "create", str(in_img), "-o", str(out_vtf), "-f", str(fmt), "--no-pretty-formatting", "--verbose"]
    if srgb:
        args.append("--srgb")
    for fl in flags:
        args += ["--flag", fl]
    log_fn("MareTF: " + " ".join(args))
    cp = subprocess.run(args, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError((cp.stdout or "") + "\n" + (cp.stderr or ""))

def try_maretf_create(maretf_path: str, in_img: Path, out_vtf: Path, fmts, srgb: bool, flags, log_fn):
    last_err = None
    for fmt in fmts:
        try:
            maretf_create_vtf(maretf_path, in_img, out_vtf, fmt, srgb, flags, log_fn)
            return fmt
        except Exception as e:
            last_err = e
    raise RuntimeError(str(last_err) if last_err else "MareTF create a échoué.")

def pick_color_formats(preset: str, has_alpha: bool):
    if preset == "optimized":
        return ["DXT5"] if has_alpha else ["DXT1"]
    if preset == "super_optimized":
        return ["DXT5"] if has_alpha else ["DXT1"]
    if preset == "uncompressed":
        return ["BGRA8888"] if has_alpha else ["BGR888", "RGB888"]
    return ["DXT5"] if has_alpha else ["DXT1"]

def pick_normal_formats(preset: str, needs_alpha: bool):
    if preset == "uncompressed":
        return ["BGRA8888"] if needs_alpha else ["BGR888", "RGB888"]
    return ["DXT5"]

def pick_gray_formats(preset: str):
    if preset == "uncompressed":
        return ["BGR888", "RGB888"]
    return ["DXT1"]

def open_folder(path: Path):
    try:
        p = str(path)
        if os.name == "nt":
            subprocess.Popen(["explorer", p])
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception:
        pass

class FileRow(QWidget):
    def __init__(self, label: str, is_dir=False, parent=None):
        super().__init__(parent)
        self.is_dir = is_dir
        self.lab = QLabel(label)
        self.edit = QLineEdit()
        self.btn = QPushButton("Parcourir")
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.lab)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)
        self.setLayout(lay)
        self.btn.clicked.connect(self.browse)

    def browse(self):
        if self.is_dir:
            p = QFileDialog.getExistingDirectory(self, "Choisir un dossier", self.edit.text() or "")
            if p:
                self.edit.setText(p)
        else:
            p, _ = QFileDialog.getOpenFileName(
                self,
                "Choisir un fichier",
                self.edit.text() or "",
                "Images/VTF (*.png *.tga *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.vtf);;Tous (*.*)"
            )
            if p:
                self.edit.setText(p)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_enabled(self, on: bool):
        self.setVisible(on)
        self.setEnabled(on)

class Worker(QObject):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(str)
    error = Signal(str)

    def __init__(self, cfg: dict, cancel_event: threading.Event):
        super().__init__()
        self.cfg = cfg
        self.cancel_event = cancel_event

    def _log(self, s: str):
        self.log.emit(s)

    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise RuntimeError("Annulé")

    def _step(self, p: int, msg: str):
        self._check_cancel()
        self.progress.emit(p)
        self._log(msg)

    def run(self):
        created_outputs = []
        try:
            cfg = self.cfg
            profile = cfg["profile"]
            maretf = cfg["maretf"].strip()
            preset = cfg["preset"]
            out_root = Path(cfg["out_root"])
            mat_path = cfg["mat_path"].strip().replace("\\", "/").strip("/")

            if not maretf:
                raise RuntimeError("MareTF.exe requis.")
            if not out_root.exists():
                raise RuntimeError("Output root invalide.")
            if not mat_path:
                raise RuntimeError("Material path requis.")

            target_dir = out_root / mat_path
            target_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory(prefix="pbrmerge_") as td:
                tmp = Path(td)

                def load_as_file(p: str, tag: str) -> Path:
                    self._check_cancel()
                    src = Path(p)
                    if is_vtf(p):
                        out_png = tmp / f"{src.stem}_{tag}.png"
                        maretf_extract_to_png(maretf, src, out_png, self._log)
                        return out_png
                    return src

                if profile == "vanilla":
                    base_path = cfg["van_basecolor"].strip()
                    normal_path = cfg["van_normal"].strip()
                    phong_src_path = cfg["van_phong_src"].strip()

                    lvl_black = cfg["van_lvl_black"]
                    lvl_gamma = cfg["van_lvl_gamma"]
                    lvl_white = cfg["van_lvl_white"]

                    phongboost = cfg["van_phongboost"]
                    env_enabled = cfg["van_env_enabled"]
                    env_r = cfg["van_env_r"]
                    env_g = cfg["van_env_g"]
                    env_b = cfg["van_env_b"]

                    emissive_noise = cfg["van_emissive_noise"]
                    noise_min = cfg["van_noise_min"]
                    noise_max = cfg["van_noise_max"]
                    emissive_tex_path = cfg["van_emissive_tex"].strip()

                    color2_r = cfg["van_color2_r"]
                    color2_g = cfg["van_color2_g"]
                    color2_b = cfg["van_color2_b"]

                    if not base_path:
                        raise RuntimeError("Vanilla: Basecolor requis.")
                    if not normal_path:
                        raise RuntimeError("Vanilla: Normal requis.")
                    if not phong_src_path:
                        raise RuntimeError("Vanilla: Texture phong exponent (source) requise.")
                    if emissive_noise and not emissive_tex_path:
                        raise RuntimeError("Vanilla: Emissive noise activé -> texture emissive requise.")

                    name = Path(base_path).stem

                    vtf_base = target_dir / f"{name}.vtf"
                    vtf_nm = target_dir / f"{name}_nm.vtf"
                    vtf_g = target_dir / f"{name}_g.vtf"
                    vtf_em = target_dir / f"{name}_em.vtf"
                    vmt_path = target_dir / f"{name}.vmt"
                    created_outputs += [vtf_base, vtf_nm, vtf_g, vmt_path]
                    if emissive_noise:
                        created_outputs.append(vtf_em)

                    self._step(10, "Chargement basecolor...")
                    base_src = load_as_file(base_path, "vbase")
                    base_im, has_alpha = img_open_rgba_if_any(str(base_src))
                    size = base_im.size

                    self._step(18, "Préparation basecolor...")
                    base_png = tmp / f"{name}.png"
                    base_im.save(base_png)

                    self._step(26, "Préparation phong exponent (greyscale + levels)...")
                    ph_src = load_as_file(phong_src_path, "vph")
                    ph_l = resize_to(img_open_luma_8bit(str(ph_src)), size)
                    ph_l2 = apply_levels_l_8bit(ph_l, lvl_black, lvl_gamma, lvl_white)

                    self._step(34, "Préparation $phongexponenttexture (R=phong, G=0, B=0)...")
                    z = Image.new("L", size, 0)
                    ph_rgb = Image.merge("RGB", (ph_l2, z, z))
                    ph_png = tmp / f"{name}_g.png"
                    ph_rgb.save(ph_png)

                    self._step(44, "Préparation normal (alpha = phong greyscale)...")
                    nrm_src = load_as_file(normal_path, "vnm")
                    nrm_rgb = resize_to(img_open_rgb(str(nrm_src)), size)
                    nrm_rgba = nrm_rgb.convert("RGBA")
                    nrm_rgba.putalpha(ph_l2)
                    nrm_png = tmp / f"{name}_nm.png"
                    nrm_rgba.save(nrm_png)

                    em_png = None
                    if emissive_noise:
                        self._step(54, "Préparation emissive texture...")
                        em_src = load_as_file(emissive_tex_path, "vem")
                        em_rgb = resize_to(img_open_rgb(str(em_src)), size)
                        em_png = tmp / f"{name}_em.png"
                        em_rgb.save(em_png)

                    self._step(62, "Écriture VMT...")
                    base_ref = f"{mat_path}/{name}".replace("\\", "/")
                    nm_ref = f"{mat_path}/{name}_nm".replace("\\", "/")
                    g_ref = f"{mat_path}/{name}_g".replace("\\", "/")
                    em_ref = f"{mat_path}/{name}_em".replace("\\", "/")

                    vmt_lines = []
                    vmt_lines.append('"VertexlitGeneric"')
                    vmt_lines.append("{")
                    vmt_lines.append(f'\t"$basetexture"\t\t\t\t\t"{base_ref}"')
                    vmt_lines.append(f'\t"$bumpmap"\t\t\t\t\t\t"{nm_ref}"')
                    vmt_lines.append(f'\t"$phongexponenttexture"\t\t"{g_ref}"')
                    vmt_lines.append("")
                    if has_alpha:
                        vmt_lines.append(f'\t"$color2"\t"[{color2_r:.2f} {color2_g:.2f} {color2_b:.2f}]"')
                        vmt_lines.append('\t"$translucent"\t"1"')
                        vmt_lines.append("")
                    vmt_lines.append('\t"$phong"\t\t\t\t\t\t"1"')
                    vmt_lines.append(f'\t"$phongboost"\t\t\t\t"{phongboost:.3f}"')
                    vmt_lines.append('\t"$phongfresnelranges"\t\t"[0.015 0.1 0.5]"')
                    vmt_lines.append('\t"$PhongDisableHalfLambert"\t\t"1"')
                    vmt_lines.append("")
                    vmt_lines.append('\t"$normalmapalphaenvmapmask"\t"1"')

                    if env_enabled:
                        vmt_lines.append('\t"$envmap"\t"env_cubemap"')
                        vmt_lines.append(f'\t"$envmaptint"\t"[{env_r:.3f} {env_g:.3f} {env_b:.3f}]"')
                        vmt_lines.append("")

                    if emissive_noise:
                        vmt_lines.append('\t"$emissiveBlendEnabled"\t"1"')
                        vmt_lines.append(f'\t"$emissiveBlendTexture"\t"{em_ref}"')
                        vmt_lines.append('\t"$emissiveBlendFlowTexture"\t"dev/null"')
                        vmt_lines.append('\t"$emissiveBlendStrength"\t"1"')
                        vmt_lines.append('\t"$emissiveBlendScrollVector"\t"[0 0]"')
                        vmt_lines.append("")
                        vmt_lines.append('\t"Proxies"')
                        vmt_lines.append("\t{")
                        vmt_lines.append('\t\t"UniformNoise"')
                        vmt_lines.append("\t\t{")
                        vmt_lines.append('\t\t\t"resultVar"\t"$emissiveBlendTint"')
                        vmt_lines.append(f'\t\t\t"minval"\t{noise_min:.3f}')
                        vmt_lines.append(f'\t\t\t"maxval"\t{noise_max:.3f}')
                        vmt_lines.append("\t\t}")
                        vmt_lines.append("\t}")
                        vmt_lines.append("")

                    vmt_lines.append("}")
                    vmt_path.write_text("\n".join(vmt_lines), encoding="utf-8")

                    self._step(78, "Compilation VTF base...")
                    fmt_base = try_maretf_create(maretf, base_png, vtf_base, pick_color_formats(preset, has_alpha), True, [], self._log)
                    self._log(f"VTF base format: {fmt_base} (alpha={'YES' if has_alpha else 'NO'})")

                    self._step(88, "Compilation VTF normal (avec alpha)...")
                    fmt_nm = try_maretf_create(maretf, nrm_png, vtf_nm, pick_normal_formats(preset, True), False, ["NORMAL"], self._log)
                    self._log(f"VTF normal format: {fmt_nm}")

                    self._step(94, "Compilation VTF phong exponent...")
                    fmt_g = try_maretf_create(maretf, ph_png, vtf_g, pick_gray_formats(preset), False, [], self._log)
                    self._log(f"VTF phong exponent format: {fmt_g}")

                    if emissive_noise and em_png:
                        self._step(98, "Compilation VTF emissive...")
                        fmt_em = try_maretf_create(maretf, em_png, vtf_em, pick_color_formats(preset, False), True, [], self._log)
                        self._log(f"VTF emissive format: {fmt_em}")

                else:
                    raise RuntimeError("Ce fichier ne contient que le profil Vanilla corrigé (env/emissive/phong+alpha normal).")

            self._step(100, "Terminé.")
            self.done.emit(str(target_dir))
        except Exception as e:
            for p in created_outputs:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PBR Merger GMod (Vanilla)")
        self.settings = QSettings("PBRMergerGMod", "PBRMergerGMod")
        self.cancel_event = threading.Event()
        self.profile = "vanilla"

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_select = QWidget()
        self.page_main = QWidget()
        self.stack.addWidget(self.page_select)
        self.stack.addWidget(self.page_main)

        self._build_select_page()
        self._build_main_page()

        self._load_settings()
        self._apply_ui()

        self.resize(520, 200)
        self.stack.setCurrentWidget(self.page_select)

    def _build_select_page(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        self.page_select.setLayout(lay)

        title = QLabel("Choisir un profil")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        title.setStyleSheet("font-size: 18px;")
        lay.addWidget(title)

        btns = QHBoxLayout()
        btns.setSpacing(12)
        lay.addLayout(btns)

        self.btn_vanilla = QPushButton("Vanilla (Phong)")
        self.btn_vanilla.setMinimumHeight(70)
        btns.addWidget(self.btn_vanilla)

        self.btn_vanilla.clicked.connect(lambda: self._enter_profile("vanilla"))

    def _build_main_page(self):
        root = self.page_main
        main = QVBoxLayout()
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)
        root.setLayout(main)

        top = QGridLayout()
        top.setHorizontalSpacing(10)
        top.setVerticalSpacing(8)
        main.addLayout(top)

        self.back_btn = QPushButton("Retour")
        top.addWidget(self.back_btn, 0, 0)

        self.preset = QComboBox()
        self.preset.addItems(["optimized", "super_optimized", "uncompressed"])
        top.addWidget(QLabel("Preset VTF"), 0, 1)
        top.addWidget(self.preset, 0, 2)

        self.maretf = FileRow("MareTF.exe")
        self.out_root = FileRow("Dossier output (materials)", is_dir=True)
        self.mat_path = QLineEdit()
        self.mat_path.setPlaceholderText("ex: models/tonpack/monmat")

        top.addWidget(QLabel("Chemin MareTF"), 1, 0)
        top.addWidget(self.maretf, 1, 1, 1, 2)
        top.addWidget(QLabel("Output root"), 2, 0)
        top.addWidget(self.out_root, 2, 1, 1, 2)
        top.addWidget(QLabel("Material path"), 3, 0)
        top.addWidget(self.mat_path, 3, 1, 1, 2)

        self.van_basecolor = FileRow("Basecolor (RGB/RGBA ou VTF)")
        self.van_normal = FileRow("Normal (RGB/VTF)")
        self.van_phong_src = FileRow("Phong exponent source (greyscale) (L/RGB/VTF)")
        main.addWidget(self.van_basecolor)
        main.addWidget(self.van_normal)
        main.addWidget(self.van_phong_src)

        van_lvl = QHBoxLayout()
        self.van_apply_levels = QCheckBox("Levels Phong (Noir/Gamma/Blanc)")
        self.van_apply_levels.setChecked(True)
        self.van_black = QSpinBox()
        self.van_black.setRange(0, 255)
        self.van_black.setValue(255)
        self.van_gamma = QDoubleSpinBox()
        self.van_gamma.setRange(0.01, 9.99)
        self.van_gamma.setSingleStep(0.05)
        self.van_gamma.setValue(0.35)
        self.van_white = QSpinBox()
        self.van_white.setRange(0, 255)
        self.van_white.setValue(255)
        van_lvl.addWidget(self.van_apply_levels)
        van_lvl.addWidget(QLabel("Noir"))
        van_lvl.addWidget(self.van_black)
        van_lvl.addWidget(QLabel("Gamma"))
        van_lvl.addWidget(self.van_gamma)
        van_lvl.addWidget(QLabel("Blanc"))
        van_lvl.addWidget(self.van_white)
        van_lvl.addStretch(1)
        main.addLayout(van_lvl)

        van_params = QHBoxLayout()
        self.van_phongboost = QDoubleSpinBox()
        self.van_phongboost.setRange(0.0, 200.0)
        self.van_phongboost.setSingleStep(1.0)
        self.van_phongboost.setValue(25.0)

        self.van_env_enabled = QCheckBox("Activer envmap")
        self.van_env_enabled.setChecked(True)

        self.van_env_r = QDoubleSpinBox()
        self.van_env_g = QDoubleSpinBox()
        self.van_env_b = QDoubleSpinBox()
        for sp in (self.van_env_r, self.van_env_g, self.van_env_b):
            sp.setRange(0.0, 2.0)
            sp.setSingleStep(0.05)
            sp.setValue(0.1)

        van_params.addWidget(QLabel("Phong boost"))
        van_params.addWidget(self.van_phongboost)
        van_params.addSpacing(18)
        van_params.addWidget(self.van_env_enabled)
        van_params.addWidget(QLabel("Env tint"))
        van_params.addWidget(self.van_env_r)
        van_params.addWidget(self.van_env_g)
        van_params.addWidget(self.van_env_b)
        van_params.addStretch(1)
        main.addLayout(van_params)

        van_col = QHBoxLayout()
        self.van_color2_r = QDoubleSpinBox()
        self.van_color2_g = QDoubleSpinBox()
        self.van_color2_b = QDoubleSpinBox()
        for sp in (self.van_color2_r, self.van_color2_g, self.van_color2_b):
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.05)
            sp.setValue(0.85)
        van_col.addWidget(QLabel("Color2 (si alpha base)"))
        van_col.addWidget(self.van_color2_r)
        van_col.addWidget(self.van_color2_g)
        van_col.addWidget(self.van_color2_b)
        van_col.addStretch(1)
        main.addLayout(van_col)

        van_emi = QHBoxLayout()
        self.van_emissive_noise = QCheckBox("Emissive noise (UniformNoise proxy)")
        self.van_emissive_noise.setChecked(False)
        self.van_emissive_tex = FileRow("Emissive texture (RGB/VTF)")
        self.van_noise_min = QDoubleSpinBox()
        self.van_noise_max = QDoubleSpinBox()
        for sp in (self.van_noise_min, self.van_noise_max):
            sp.setRange(0.0, 10.0)
            sp.setSingleStep(0.05)
        self.van_noise_min.setValue(0.5)
        self.van_noise_max.setValue(1.0)
        van_emi.addWidget(self.van_emissive_noise)
        van_emi.addWidget(QLabel("min"))
        van_emi.addWidget(self.van_noise_min)
        van_emi.addWidget(QLabel("max"))
        van_emi.addWidget(self.van_noise_max)
        van_emi.addStretch(1)
        main.addLayout(van_emi)
        main.addWidget(self.van_emissive_tex)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Générer (VMT + VTF)")
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setEnabled(False)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        main.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        main.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        main.addWidget(self.log, 1)

        self.back_btn.clicked.connect(self._go_back)
        self.run_btn.clicked.connect(self._start_job)
        self.cancel_btn.clicked.connect(self._cancel_job)
        self.van_emissive_noise.stateChanged.connect(self._apply_ui)
        self.van_env_enabled.stateChanged.connect(self._apply_ui)

    def _enter_profile(self, prof: str):
        self.profile = prof
        self.resize(1100, 900)
        self.stack.setCurrentWidget(self.page_main)
        self._apply_ui()

    def _go_back(self):
        if self.cancel_btn.isEnabled():
            return
        self._save_settings()
        self.resize(520, 200)
        self.stack.setCurrentWidget(self.page_select)

    def _apply_ui(self):
        emi_on = self.van_emissive_noise.isChecked()
        self.van_emissive_tex.set_enabled(emi_on)
        self.van_noise_min.setEnabled(emi_on)
        self.van_noise_max.setEnabled(emi_on)

        env_on = self.van_env_enabled.isChecked()
        self.van_env_r.setEnabled(env_on)
        self.van_env_g.setEnabled(env_on)
        self.van_env_b.setEnabled(env_on)

    def _cfg(self):
        apply = self.van_apply_levels.isChecked()
        b = int(self.van_black.value())
        g = float(self.van_gamma.value())
        w = int(self.van_white.value())

        return {
            "profile": "vanilla",
            "maretf": self.maretf.path(),
            "out_root": self.out_root.path(),
            "mat_path": self.mat_path.text().strip(),
            "preset": self.preset.currentText(),

            "van_basecolor": self.van_basecolor.path(),
            "van_normal": self.van_normal.path(),
            "van_phong_src": self.van_phong_src.path(),

            "van_lvl_black": b if apply else 0,
            "van_lvl_gamma": g if apply else 1.0,
            "van_lvl_white": w if apply else 255,

            "van_phongboost": float(self.van_phongboost.value()),
            "van_env_enabled": self.van_env_enabled.isChecked(),
            "van_env_r": float(self.van_env_r.value()),
            "van_env_g": float(self.van_env_g.value()),
            "van_env_b": float(self.van_env_b.value()),

            "van_color2_r": float(self.van_color2_r.value()),
            "van_color2_g": float(self.van_color2_g.value()),
            "van_color2_b": float(self.van_color2_b.value()),

            "van_emissive_noise": self.van_emissive_noise.isChecked(),
            "van_emissive_tex": self.van_emissive_tex.path(),
            "van_noise_min": float(self.van_noise_min.value()),
            "van_noise_max": float(self.van_noise_max.value()),
        }

    def _cancel_job(self):
        self.cancel_event.set()
        self.cancel_btn.setEnabled(False)

    def _start_job(self):
        cfg = self._cfg()
        if not cfg["maretf"] or not cfg["out_root"] or not cfg["mat_path"]:
            QMessageBox.critical(self, "Erreur", "MareTF, Output root et Material path sont requis.")
            return

        if not cfg["van_basecolor"] or not cfg["van_normal"] or not cfg["van_phong_src"]:
            QMessageBox.critical(self, "Erreur", "Basecolor + Normal + Phong source requis (profil Vanilla).")
            return

        if cfg["van_emissive_noise"] and not cfg["van_emissive_tex"]:
            QMessageBox.critical(self, "Erreur", "Emissive noise activé -> texture emissive requise.")
            return

        self._save_settings()
        self.run_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_event.clear()
        self.progress.setValue(0)
        self.log.clear()

        self.thread = QThread()
        self.worker = Worker(cfg, self.cancel_event)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log.append)
        self.worker.done.connect(self._job_done)
        self.worker.error.connect(self._job_error)

        self.worker.done.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _job_done(self, outdir: str):
        self.progress.setValue(100)
        self.run_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        open_folder(Path(outdir))

    def _job_error(self, msg: str):
        self.log.append("ERREUR: " + msg)
        self.run_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _save_settings(self):
        s = self.settings
        s.setValue("maretf", self.maretf.path())
        s.setValue("out_root", self.out_root.path())
        s.setValue("mat_path", self.mat_path.text().strip())
        s.setValue("preset", self.preset.currentText())

    def _load_settings(self):
        s = self.settings
        self.maretf.edit.setText(s.value("maretf", "", str))
        self.out_root.edit.setText(s.value("out_root", "", str))
        self.mat_path.setText(s.value("mat_path", "", str))
        preset = s.value("preset", "optimized", str)
        i = self.preset.findText(preset)
        if i >= 0:
            self.preset.setCurrentIndex(i)

        self.van_basecolor.edit.setText("")
        self.van_normal.edit.setText("")
        self.van_phong_src.edit.setText("")
        self.van_emissive_tex.edit.setText("")
        self._apply_ui()

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
