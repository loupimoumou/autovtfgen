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

def fmt2(x: float) -> str:
    return f"{float(x):.2f}"

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
    if preset in ("optimized", "super_optimized"):
        return ["DXT5"] if has_alpha else ["DXT1"]
    if preset == "uncompressed":
        return ["BGRA8888"] if has_alpha else ["BGR888", "RGB888"]
    return ["DXT5"] if has_alpha else ["DXT1"]

def pick_arm_formats(preset: str, has_alpha: bool):
    if preset in ("optimized", "super_optimized"):
        return ["DXT5"] if has_alpha else ["DXT1"]
    if preset == "uncompressed":
        return ["BGRA8888"] if has_alpha else ["BGR888", "RGB888"]
    return ["DXT5"] if has_alpha else ["DXT1"]

def pick_normal_formats(preset: str, needs_alpha: bool):
    if preset == "uncompressed":
        return ["BGRA8888"] if needs_alpha else ["BGR888", "RGB888"]
    return ["DXT5"]

def pick_gray_rgb_formats(preset: str):
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

            with tempfile.TemporaryDirectory(prefix="texmerge_") as td:
                tmp = Path(td)

                def load_as_file(p: str, tag: str) -> Path:
                    self._check_cancel()
                    src = Path(p)
                    if is_vtf(p):
                        out_png = tmp / f"{src.stem}_{tag}.png"
                        maretf_extract_to_png(maretf, src, out_png, self._log)
                        return out_png
                    return src

                if profile == "metro":
                    base_path = cfg["m_basecolor"].strip()
                    illum_path = cfg["m_illum"].strip()
                    ao_path = cfg["m_ao"].strip()
                    normal_path = cfg["m_normal"].strip()
                    metallic_path = cfg["m_metallic"].strip()
                    roughness_path = cfg["m_roughness"].strip()
                    bump_path = cfg["m_bump"].strip()
                    height_path = cfg["m_height"].strip()

                    metro_mode = cfg["m_mode"]
                    inv_green = cfg["m_inv_green"]
                    apply_levels = cfg["m_apply_levels"]
                    black_in = cfg["m_black_in"]
                    gamma = cfg["m_gamma"]
                    white_in = cfg["m_white_in"]

                    parallaxscale = cfg["m_parallaxscale"]
                    selfillumscale = cfg["m_selfillumscale"]
                    global_illum = cfg["m_global_illum"]

                    if not base_path:
                        raise RuntimeError("Metro: Basecolor requis.")
                    if not normal_path:
                        raise RuntimeError("Metro: Normal requis.")
                    if metro_mode == "packed":
                        if not bump_path:
                            raise RuntimeError("Metro: Bump pack requis (G=Rough, B=Metal).")
                    else:
                        if not metallic_path or not roughness_path:
                            raise RuntimeError("Metro: Metallic + Roughness requis.")

                    name = Path(base_path).stem
                    use_rgbi = bool(illum_path) or global_illum
                    rgb_suffix = "rgbi" if use_rgbi else "rgb"

                    vtf_rgb = target_dir / f"{name}_{rgb_suffix}.vtf"
                    vtf_arm = target_dir / f"{name}_arm.vtf"
                    vtf_nrm = target_dir / f"{name}_nrm.vtf"
                    vmt_path = target_dir / f"{name}.vmt"
                    created_outputs += [vtf_rgb, vtf_arm, vtf_nrm, vmt_path]

                    self._step(8, "Chargement basecolor...")
                    base_src = load_as_file(base_path, "base")
                    base_rgb = img_open_rgb(str(base_src))
                    size = base_rgb.size

                    self._step(18, "Basecolor alpha (selfillum mask)...")
                    rgb_png = tmp / f"{name}_{rgb_suffix}.png"
                    if use_rgbi:
                        if global_illum:
                            a = Image.new("L", size, 255)
                        elif illum_path:
                            illum_src = load_as_file(illum_path, "illum")
                            a = resize_to(img_open_luma_8bit(str(illum_src)), size)
                        else:
                            a = Image.new("L", size, 0)
                        base_rgba = base_rgb.convert("RGBA")
                        base_rgba.putalpha(a)
                        base_rgba.save(rgb_png)
                    else:
                        base_rgb.save(rgb_png)

                    self._step(28, "AO...")
                    if ao_path:
                        ao_src = load_as_file(ao_path, "ao")
                        ao_l = resize_to(img_open_luma_8bit(str(ao_src)), size)
                    else:
                        ao_l = Image.new("L", size, 255)

                    self._step(40, "Rough/Metal...")
                    if metro_mode == "packed":
                        bump_src = load_as_file(bump_path, "bump")
                        bump = Image.open(str(bump_src))
                        if bump.mode not in ("RGB", "RGBA"):
                            bump = bump.convert("RGBA") if "A" in bump.getbands() else bump.convert("RGB")
                        bump = resize_to(bump, size)
                        if bump.mode == "RGBA":
                            _, gch, bch, _ = bump.split()
                        else:
                            _, gch, bch = bump.split()
                        rough_l = gch.convert("L")
                        metal_before = bch.convert("L")
                    else:
                        r_src = load_as_file(roughness_path, "rgh")
                        m_src = load_as_file(metallic_path, "met")
                        rough_l = resize_to(img_open_luma_8bit(str(r_src)), size)
                        metal_before = resize_to(img_open_luma_8bit(str(m_src)), size)

                    metal_l = metal_before
                    s_before = stats_l(metal_before)
                    if apply_levels:
                        metal_l = apply_levels_l_8bit(metal_l, black_in, gamma, white_in)
                    s_after = stats_l(metal_l)
                    d = hist_distance(metal_before, metal_l)

                    self._log("ARM packing: R=AO, G=Rough, B=Metal, A=Height(optionnel)")
                    self._log(f"Metal before: min={s_before['min']} max={s_before['max']} mean={s_before['mean']:.2f} p05={s_before['p05']} p50={s_before['p50']} p95={s_before['p95']} unique={s_before['unique']} 0={s_before['c0']} 255={s_before['c255']}")
                    self._log(f"Metal after : min={s_after['min']} max={s_after['max']} mean={s_after['mean']:.2f} p05={s_after['p05']} p50={s_after['p50']} p95={s_after['p95']} unique={s_after['unique']} 0={s_after['c0']} 255={s_after['c255']}")
                    self._log(f"Metal hist Δ (norm): {d:.4f} (levels={'ON' if apply_levels else 'OFF'} b={black_in} g={gamma} w={white_in})")

                    self._step(55, "Normal...")
                    nrm_src = load_as_file(normal_path, "nrm")
                    nrm = resize_to(img_open_rgb(str(nrm_src)), size)
                    if inv_green:
                        nrm = norm_invert_green(nrm)

                    self._step(66, "ARM + Height(optionnel)...")
                    arm_png = tmp / f"{name}_arm.png"
                    nrm_png = tmp / f"{name}_nrm.png"

                    if height_path:
                        h_src = load_as_file(height_path, "hgt")
                        h_l = resize_to(img_open_luma_8bit(str(h_src)), size)
                        arm_rgba = Image.merge("RGBA", (ao_l, rough_l, metal_l, h_l))
                        arm_rgba.save(arm_png)
                        arm_has_alpha = True
                    else:
                        arm_rgb = Image.merge("RGB", (ao_l, rough_l, metal_l))
                        arm_rgb.save(arm_png)
                        arm_has_alpha = False

                    nrm.save(nrm_png)

                    self._step(74, "Écriture VMT (ExoPBR)...")
                    base_ref = f"{mat_path}/{name}_{rgb_suffix}".replace("\\", "/")
                    arm_ref = f"{mat_path}/{name}_arm".replace("\\", "/")
                    nrm_ref = f"{mat_path}/{name}_nrm".replace("\\", "/")

                    extra = ""
                    if use_rgbi:
                        extra += f'\t$selfillumscale "{selfillumscale:.3f}"\n'
                    if parallaxscale > 0.0:
                        extra += f'\t$parallaxscale "{parallaxscale:.3f}"\n'

                    vmt = (
                        "screenspace_general\n"
                        "{\n"
                        '\t$pixshader "exopbr1_standard_ps30"\n'
                        '\t$vertexshader "exopbr1_standard_vs30"\n\n'
                        f'\t$basetexture "{base_ref}"\n'
                        f'\t$texture1 "{arm_ref}"\n'
                        f'\t$texture2 "{nrm_ref}"\n'
                        f"{extra}\n"
                        "\t$model 1\n\n"
                        "\tProxies {\n"
                        "\t\tExoPBR {}\n"
                        "\t}\n"
                        "}\n"
                    )
                    vmt_path.write_text(vmt, encoding="utf-8")

                    self._step(84, "Compilation VTF basecolor...")
                    fmt_base = try_maretf_create(maretf, rgb_png, vtf_rgb, pick_color_formats(preset, use_rgbi), True, [], self._log)
                    self._log(f"VTF base format: {fmt_base} (alpha={'YES' if use_rgbi else 'NO'})")

                    self._step(92, "Compilation VTF ARM...")
                    fmt_arm = try_maretf_create(maretf, arm_png, vtf_arm, pick_arm_formats(preset, arm_has_alpha), False, [], self._log)
                    self._log(f"VTF ARM format: {fmt_arm} (alpha={'YES' if arm_has_alpha else 'NO'})")

                    self._step(98, "Compilation VTF NRM...")
                    fmt_nrm = try_maretf_create(maretf, nrm_png, vtf_nrm, pick_normal_formats(preset, False), False, ["NORMAL"], self._log)
                    self._log(f"VTF NRM format: {fmt_nrm}")

                elif profile == "vanilla":
                    base_path = cfg["v_basecolor"].strip()
                    normal_path = cfg["v_normal"].strip()
                    phong_src_path = cfg["v_phong_src"].strip()

                    trans_enable = cfg["v_trans_enable"]
                    trans_mask_path = cfg["v_trans_mask"].strip()

                    lvl_black = cfg["v_lvl_black"]
                    lvl_gamma = cfg["v_lvl_gamma"]
                    lvl_white = cfg["v_lvl_white"]

                    phongboost = cfg["v_phongboost"]
                    env_enabled = cfg["v_env_enabled"]
                    env_r = cfg["v_env_r"]
                    env_g = cfg["v_env_g"]
                    env_b = cfg["v_env_b"]

                    emissive_noise = cfg["v_emissive_noise"]
                    emissive_tex_path = cfg["v_emissive_tex"].strip()
                    noise_min = cfg["v_noise_min"]
                    noise_max = cfg["v_noise_max"]

                    color2_r = cfg["v_color2_r"]
                    color2_g = cfg["v_color2_g"]
                    color2_b = cfg["v_color2_b"]

                    if not base_path:
                        raise RuntimeError("Vanilla: Basecolor requis.")
                    if not normal_path:
                        raise RuntimeError("Vanilla: Normal requis.")
                    if not phong_src_path:
                        raise RuntimeError("Vanilla: Texture phong exponent source requise.")
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
                    base_im, base_has_alpha = img_open_rgba_if_any(str(base_src))
                    size = base_im.size

                    self._step(18, "Transparence (optionnelle)...")
                    has_alpha_out = False
                    base_out = None

                    if trans_enable:
                        if trans_mask_path:
                            mask_src = load_as_file(trans_mask_path, "vtr")
                            alpha_l = resize_to(img_open_luma_8bit(str(mask_src)), size)
                            base_rgb = base_im.convert("RGB")
                            base_out = base_rgb.convert("RGBA")
                            base_out.putalpha(alpha_l)
                            has_alpha_out = True
                        else:
                            if base_has_alpha:
                                base_out = base_im.convert("RGBA")
                                has_alpha_out = True
                            else:
                                base_out = base_im.convert("RGB")
                                has_alpha_out = False
                    else:
                        base_out = base_im.convert("RGB")
                        has_alpha_out = False

                    base_png = tmp / f"{name}.png"
                    base_out.save(base_png)

                    self._step(26, "Phong exponent (greyscale + levels)...")
                    ph_src = load_as_file(phong_src_path, "vph")
                    ph_l = resize_to(img_open_luma_8bit(str(ph_src)), size)
                    ph_l2 = apply_levels_l_8bit(ph_l, lvl_black, lvl_gamma, lvl_white)

                    self._step(34, "Texture phongexponent (R=phong, G=0, B=0)...")
                    z = Image.new("L", size, 0)
                    ph_rgb = Image.merge("RGB", (ph_l2, z, z))
                    ph_png = tmp / f"{name}_g.png"
                    ph_rgb.save(ph_png)

                    self._step(44, "Normal alpha = phong...")
                    nrm_src = load_as_file(normal_path, "vnm")
                    nrm_rgb = resize_to(img_open_rgb(str(nrm_src)), size)
                    nrm_rgba = nrm_rgb.convert("RGBA")
                    nrm_rgba.putalpha(ph_l2)
                    nrm_png = tmp / f"{name}_nm.png"
                    nrm_rgba.save(nrm_png)

                    em_png = None
                    if emissive_noise:
                        self._step(54, "Emissive texture...")
                        em_src = load_as_file(emissive_tex_path, "vem")
                        em_rgb = resize_to(img_open_rgb(str(em_src)), size)
                        em_png = tmp / f"{name}_em.png"
                        em_rgb.save(em_png)

                    self._step(64, "Écriture VMT (Phong)...")
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

                    if trans_enable and has_alpha_out:
                        vmt_lines.append(f'\t"$color2"\t"[{fmt2(color2_r)} {fmt2(color2_g)} {fmt2(color2_b)}]"')
                        vmt_lines.append('\t"$translucent"\t"1"')
                        vmt_lines.append("")

                    vmt_lines.append('\t"$phong"\t\t\t\t\t\t"1"')
                    vmt_lines.append(f'\t"$phongboost"\t\t\t\t"{int(phongboost)}"')
                    vmt_lines.append('\t"$phongfresnelranges"\t\t"[0.015 0.1 0.5]"')
                    vmt_lines.append('\t"$PhongDisableHalfLambert"\t\t"1"')
                    vmt_lines.append("")
                    vmt_lines.append('\t"$normalmapalphaenvmapmask"\t"1"')

                    if env_enabled:
                        vmt_lines.append('\t"$envmap"\t"env_cubemap"')
                        vmt_lines.append(f'\t"$envmaptint"\t"[{fmt2(env_r)} {fmt2(env_g)} {fmt2(env_b)}]"')
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
                        vmt_lines.append(f'\t\t\t"minval"\t{fmt2(noise_min)}')
                        vmt_lines.append(f'\t\t\t"maxval"\t{fmt2(noise_max)}')
                        vmt_lines.append("\t\t}")
                        vmt_lines.append("\t}")
                        vmt_lines.append("")

                    vmt_lines.append("}")
                    vmt_path.write_text("\n".join(vmt_lines), encoding="utf-8")

                    self._step(78, "Compilation VTF base...")
                    fmt_base = try_maretf_create(maretf, base_png, vtf_base, pick_color_formats(preset, has_alpha_out), True, [], self._log)
                    self._log(f"VTF base format: {fmt_base} (alpha={'YES' if has_alpha_out else 'NO'})")

                    self._step(88, "Compilation VTF normal (avec alpha)...")
                    fmt_nm = try_maretf_create(maretf, nrm_png, vtf_nm, pick_normal_formats(preset, True), False, ["NORMAL"], self._log)
                    self._log(f"VTF normal format: {fmt_nm}")

                    self._step(94, "Compilation VTF phong exponent...")
                    fmt_g = try_maretf_create(maretf, ph_png, vtf_g, pick_gray_rgb_formats(preset), False, [], self._log)
                    self._log(f"VTF phong exponent format: {fmt_g}")

                    if emissive_noise and em_png:
                        self._step(98, "Compilation VTF emissive...")
                        fmt_em = try_maretf_create(maretf, em_png, vtf_em, pick_color_formats(preset, False), True, [], self._log)
                        self._log(f"VTF emissive format: {fmt_em}")

                else:
                    raise RuntimeError("Profil inconnu.")

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
        self.setWindowTitle("Texture Builder (Metro PBR + Vanilla Phong)")
        self.settings = QSettings("TexBuilderGMod", "TexBuilderGMod")
        self.cancel_event = threading.Event()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_select = QWidget()
        self.page_main = QWidget()
        self.stack.addWidget(self.page_select)
        self.stack.addWidget(self.page_main)

        self.profile = "metro"

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

        self.btn_metro = QPushButton("Metro (ExoPBR)")
        self.btn_metro.setMinimumHeight(70)
        btns.addWidget(self.btn_metro)

        self.btn_vanilla = QPushButton("Vanilla (Phong)")
        self.btn_vanilla.setMinimumHeight(70)
        btns.addWidget(self.btn_vanilla)

        self.btn_metro.clicked.connect(lambda: self._enter_profile("metro"))
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

        self.profile_lab = QLabel("Profil: Metro")
        top.addWidget(self.profile_lab, 0, 1)

        self.preset = QComboBox()
        self.preset.addItems(["optimized", "super_optimized", "uncompressed"])
        top.addWidget(QLabel("Preset VTF"), 0, 2)
        top.addWidget(self.preset, 0, 3)

        self.maretf = FileRow("MareTF.exe")
        self.out_root = FileRow("Dossier output (materials)", is_dir=True)
        self.mat_path = QLineEdit()
        self.mat_path.setPlaceholderText("ex: models/tonpack/monmat")

        top.addWidget(QLabel("Chemin MareTF"), 1, 0)
        top.addWidget(self.maretf, 1, 1, 1, 3)
        top.addWidget(QLabel("Output root"), 2, 0)
        top.addWidget(self.out_root, 2, 1, 1, 3)
        top.addWidget(QLabel("Material path"), 3, 0)
        top.addWidget(self.mat_path, 3, 1, 1, 3)

        self.profile_stack = QStackedWidget()
        main.addWidget(self.profile_stack, 1)

        self.metro_panel = QWidget()
        self.vanilla_panel = QWidget()
        self.profile_stack.addWidget(self.metro_panel)
        self.profile_stack.addWidget(self.vanilla_panel)

        self._build_metro_panel()
        self._build_vanilla_panel()

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
        main.addWidget(self.log, 2)

        self.back_btn.clicked.connect(self._go_back)
        self.run_btn.clicked.connect(self._start_job)
        self.cancel_btn.clicked.connect(self._cancel_job)

        self.m_mode.currentIndexChanged.connect(self._apply_ui)
        self.m_global_illum.stateChanged.connect(self._apply_ui)
        self.v_trans_enable.stateChanged.connect(self._apply_ui)
        self.v_emissive_noise.stateChanged.connect(self._apply_ui)
        self.v_env_enabled.stateChanged.connect(self._apply_ui)

    def _build_metro_panel(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.metro_panel.setLayout(lay)

        self.m_mode = QComboBox()
        self.m_mode.addItems([
            "Utiliser bump pack (G=Rough, B=Metal)",
            "Mettre les matériaux à la main (Metal + Rough)"
        ])
        lay.addWidget(self.m_mode)

        self.m_basecolor = FileRow("Basecolor (RGB ou VTF)")
        self.m_global_illum = QCheckBox("Global Illum (alpha basecolor = 255)")
        self.m_illum = FileRow("Illum mask (L/RGB/VTF) -> alpha basecolor (optionnel)")
        self.m_ao = FileRow("AO (L/RGB/VTF) (optionnel, sinon blanc)")
        self.m_normal = FileRow("Normal (RGB/VTF) (DirectX Y-)")
        self.m_bump = FileRow("Metro bump pack (RGBA/VTF) (G=Rough, B=Metal)")
        self.m_metallic = FileRow("Metallic (L/RGB/VTF)")
        self.m_roughness = FileRow("Roughness (L/RGB/VTF)")
        self.m_height = FileRow("Height (Parallax) (L/RGB/VTF) (optionnel -> alpha ARM)")

        lay.addWidget(self.m_basecolor)
        lay.addWidget(self.m_global_illum)
        lay.addWidget(self.m_illum)
        lay.addWidget(self.m_ao)
        lay.addWidget(self.m_normal)
        lay.addWidget(self.m_bump)
        lay.addWidget(self.m_metallic)
        lay.addWidget(self.m_roughness)
        lay.addWidget(self.m_height)

        lvl = QHBoxLayout()
        self.m_apply_levels = QCheckBox("Appliquer Levels sur Metallic")
        self.m_apply_levels.setChecked(True)
        self.m_black_in = QSpinBox()
        self.m_black_in.setRange(0, 255)
        self.m_black_in.setValue(100)
        self.m_gamma = QDoubleSpinBox()
        self.m_gamma.setRange(0.01, 9.99)
        self.m_gamma.setDecimals(2)
        self.m_gamma.setSingleStep(0.05)
        self.m_gamma.setValue(1)
        self.m_white_in = QSpinBox()
        self.m_white_in.setRange(0, 255)
        self.m_white_in.setValue(255)
        lvl.addWidget(self.m_apply_levels)
        lvl.addWidget(QLabel("Noir"))
        lvl.addWidget(self.m_black_in)
        lvl.addWidget(QLabel("Gamma"))
        lvl.addWidget(self.m_gamma)
        lvl.addWidget(QLabel("Blanc"))
        lvl.addWidget(self.m_white_in)
        lvl.addStretch(1)
        lay.addLayout(lvl)

        opts = QHBoxLayout()
        self.m_inv_green = QCheckBox("Inverser le vert (Normal Y)")
        opts.addWidget(self.m_inv_green)
        opts.addStretch(1)
        lay.addLayout(opts)

        pbl = QHBoxLayout()
        self.m_parallaxscale = QDoubleSpinBox()
        self.m_parallaxscale.setRange(0.0, 1.0)
        self.m_parallaxscale.setDecimals(3)
        self.m_parallaxscale.setSingleStep(0.005)
        self.m_parallaxscale.setValue(0.0)
        self.m_selfillumscale = QDoubleSpinBox()
        self.m_selfillumscale.setRange(0.0, 10.0)
        self.m_selfillumscale.setDecimals(3)
        self.m_selfillumscale.setSingleStep(0.05)
        self.m_selfillumscale.setValue(1.0)
        pbl.addWidget(QLabel("$parallaxscale"))
        pbl.addWidget(self.m_parallaxscale)
        pbl.addSpacing(18)
        pbl.addWidget(QLabel("$selfillumscale (si illum/global)"))
        pbl.addWidget(self.m_selfillumscale)
        pbl.addStretch(1)
        lay.addLayout(pbl)

    def _build_vanilla_panel(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.vanilla_panel.setLayout(lay)

        self.v_basecolor = FileRow("Basecolor (RGB/RGBA ou VTF)")
        self.v_normal = FileRow("Normal (RGB/VTF)")
        self.v_phong_src = FileRow("Phong exponent source (greyscale) (L/RGB/VTF)")
        lay.addWidget(self.v_basecolor)
        lay.addWidget(self.v_normal)
        lay.addWidget(self.v_phong_src)

        tr = QHBoxLayout()
        self.v_trans_enable = QCheckBox("Activer transparence")
        self.v_trans_enable.setChecked(False)
        tr.addWidget(self.v_trans_enable)
        tr.addStretch(1)
        lay.addLayout(tr)

        self.v_trans_mask = FileRow("Masque transparence (L/RGB/VTF) (optionnel: sinon alpha basecolor)")
        lay.addWidget(self.v_trans_mask)

        van_lvl = QHBoxLayout()
        self.v_black = QSpinBox()
        self.v_black.setRange(0, 255)
        self.v_black.setValue(0)
        self.v_gamma = QDoubleSpinBox()
        self.v_gamma.setRange(0.01, 9.99)
        self.v_gamma.setDecimals(2)
        self.v_gamma.setSingleStep(0.05)
        self.v_gamma.setValue(0.35)
        self.v_white = QSpinBox()
        self.v_white.setRange(0, 255)
        self.v_white.setValue(255)
        van_lvl.addWidget(QLabel("Levels Phong: Noir"))
        van_lvl.addWidget(self.v_black)
        van_lvl.addWidget(QLabel("Gamma"))
        van_lvl.addWidget(self.v_gamma)
        van_lvl.addWidget(QLabel("Blanc"))
        van_lvl.addWidget(self.v_white)
        van_lvl.addStretch(1)
        lay.addLayout(van_lvl)

        van_params = QHBoxLayout()
        self.v_phongboost = QSpinBox()
        self.v_phongboost.setRange(0, 200)
        self.v_phongboost.setSingleStep(1)
        self.v_phongboost.setValue(25)

        self.v_env_enabled = QCheckBox("Activer envmap")
        self.v_env_enabled.setChecked(True)

        self.v_env_r = QDoubleSpinBox()
        self.v_env_g = QDoubleSpinBox()
        self.v_env_b = QDoubleSpinBox()
        for sp in (self.v_env_r, self.v_env_g, self.v_env_b):
            sp.setRange(0.0, 2.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.05)
            sp.setValue(0.10)

        van_params.addWidget(QLabel("Phong boost"))
        van_params.addWidget(self.v_phongboost)
        van_params.addSpacing(18)
        van_params.addWidget(self.v_env_enabled)
        van_params.addWidget(QLabel("Env tint"))
        van_params.addWidget(self.v_env_r)
        van_params.addWidget(self.v_env_g)
        van_params.addWidget(self.v_env_b)
        van_params.addStretch(1)
        lay.addLayout(van_params)

        van_col = QHBoxLayout()
        self.v_color2_r = QDoubleSpinBox()
        self.v_color2_g = QDoubleSpinBox()
        self.v_color2_b = QDoubleSpinBox()
        for sp in (self.v_color2_r, self.v_color2_g, self.v_color2_b):
            sp.setRange(0.0, 1.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.05)
            sp.setValue(0.85)
        van_col.addWidget(QLabel("Color2 (si transparence)"))
        van_col.addWidget(self.v_color2_r)
        van_col.addWidget(self.v_color2_g)
        van_col.addWidget(self.v_color2_b)
        van_col.addStretch(1)
        lay.addLayout(van_col)

        van_emi = QHBoxLayout()
        self.v_emissive_noise = QCheckBox("Emissive noise (UniformNoise proxy)")
        self.v_emissive_noise.setChecked(False)
        self.v_noise_min = QDoubleSpinBox()
        self.v_noise_max = QDoubleSpinBox()
        for sp in (self.v_noise_min, self.v_noise_max):
            sp.setRange(0.0, 10.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.05)
        self.v_noise_min.setValue(0.50)
        self.v_noise_max.setValue(1.00)
        van_emi.addWidget(self.v_emissive_noise)
        van_emi.addWidget(QLabel("min"))
        van_emi.addWidget(self.v_noise_min)
        van_emi.addWidget(QLabel("max"))
        van_emi.addWidget(self.v_noise_max)
        van_emi.addStretch(1)
        lay.addLayout(van_emi)

        self.v_emissive_tex = FileRow("Emissive texture (RGB/VTF)")
        lay.addWidget(self.v_emissive_tex)

    def _enter_profile(self, prof: str):
        self.profile = prof
        self.profile_lab.setText("Profil: " + ("Metro" if prof == "metro" else "Vanilla"))
        self.profile_stack.setCurrentIndex(0 if prof == "metro" else 1)
        self.resize(1120, 920)
        self.stack.setCurrentWidget(self.page_main)
        self._apply_ui()

    def _go_back(self):
        if self.cancel_btn.isEnabled():
            return
        self._save_settings()
        self.resize(520, 200)
        self.stack.setCurrentWidget(self.page_select)

    def _apply_ui(self):
        is_metro = (self.profile == "metro")
        self.profile_stack.setCurrentIndex(0 if is_metro else 1)

        packed = (self.m_mode.currentIndex() == 0)
        self.m_bump.set_enabled(packed)
        self.m_metallic.set_enabled(not packed)
        self.m_roughness.set_enabled(not packed)

        glob = self.m_global_illum.isChecked()
        self.m_illum.set_enabled(not glob)

        trans_on = self.v_trans_enable.isChecked()
        self.v_trans_mask.set_enabled(trans_on)

        emi_on = self.v_emissive_noise.isChecked()
        self.v_emissive_tex.set_enabled(emi_on)
        self.v_noise_min.setEnabled(emi_on)
        self.v_noise_max.setEnabled(emi_on)

        env_on = self.v_env_enabled.isChecked()
        self.v_env_r.setEnabled(env_on)
        self.v_env_g.setEnabled(env_on)
        self.v_env_b.setEnabled(env_on)

    def _cfg(self):
        base = {
            "profile": self.profile,
            "maretf": self.maretf.path(),
            "out_root": self.out_root.path(),
            "mat_path": self.mat_path.text().strip(),
            "preset": self.preset.currentText(),
        }

        if self.profile == "metro":
            metro_mode = "packed" if self.m_mode.currentIndex() == 0 else "manual"
            base.update({
                "m_mode": metro_mode,
                "m_basecolor": self.m_basecolor.path(),
                "m_global_illum": self.m_global_illum.isChecked(),
                "m_illum": self.m_illum.path(),
                "m_ao": self.m_ao.path(),
                "m_normal": self.m_normal.path(),
                "m_bump": self.m_bump.path(),
                "m_metallic": self.m_metallic.path(),
                "m_roughness": self.m_roughness.path(),
                "m_height": self.m_height.path(),
                "m_inv_green": self.m_inv_green.isChecked(),
                "m_apply_levels": self.m_apply_levels.isChecked(),
                "m_black_in": int(self.m_black_in.value()),
                "m_gamma": float(self.m_gamma.value()),
                "m_white_in": int(self.m_white_in.value()),
                "m_parallaxscale": float(self.m_parallaxscale.value()),
                "m_selfillumscale": float(self.m_selfillumscale.value()),
            })
        else:
            base.update({
                "v_basecolor": self.v_basecolor.path(),
                "v_normal": self.v_normal.path(),
                "v_phong_src": self.v_phong_src.path(),
                "v_trans_enable": self.v_trans_enable.isChecked(),
                "v_trans_mask": self.v_trans_mask.path(),
                "v_lvl_black": int(self.v_black.value()),
                "v_lvl_gamma": float(self.v_gamma.value()),
                "v_lvl_white": int(self.v_white.value()),
                "v_phongboost": int(self.v_phongboost.value()),
                "v_env_enabled": self.v_env_enabled.isChecked(),
                "v_env_r": float(self.v_env_r.value()),
                "v_env_g": float(self.v_env_g.value()),
                "v_env_b": float(self.v_env_b.value()),
                "v_color2_r": float(self.v_color2_r.value()),
                "v_color2_g": float(self.v_color2_g.value()),
                "v_color2_b": float(self.v_color2_b.value()),
                "v_emissive_noise": self.v_emissive_noise.isChecked(),
                "v_emissive_tex": self.v_emissive_tex.path(),
                "v_noise_min": float(self.v_noise_min.value()),
                "v_noise_max": float(self.v_noise_max.value()),
            })
        return base

    def _cancel_job(self):
        self.cancel_event.set()
        self.cancel_btn.setEnabled(False)

    def _start_job(self):
        cfg = self._cfg()
        if not cfg["maretf"] or not cfg["out_root"] or not cfg["mat_path"]:
            QMessageBox.critical(self, "Erreur", "MareTF, Output root et Material path sont requis.")
            return

        if cfg["profile"] == "metro":
            if not cfg["m_basecolor"] or not cfg["m_normal"]:
                QMessageBox.critical(self, "Erreur", "Metro: Basecolor + Normal requis.")
                return
            if cfg["m_mode"] == "packed" and not cfg["m_bump"]:
                QMessageBox.critical(self, "Erreur", "Metro: Bump pack requis en mode packed.")
                return
            if cfg["m_mode"] == "manual" and (not cfg["m_metallic"] or not cfg["m_roughness"]):
                QMessageBox.critical(self, "Erreur", "Metro: Metallic + Roughness requis en mode manuel.")
                return
        else:
            if not cfg["v_basecolor"] or not cfg["v_normal"] or not cfg["v_phong_src"]:
                QMessageBox.critical(self, "Erreur", "Vanilla: Basecolor + Normal + Phong source requis.")
                return
            if cfg["v_emissive_noise"] and not cfg["v_emissive_tex"]:
                QMessageBox.critical(self, "Erreur", "Vanilla: Emissive noise activé -> texture emissive requise.")
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

    def _load_settings(self):
        s = self.settings
        self.maretf.edit.setText(s.value("maretf", "", str))
        self.out_root.edit.setText(s.value("out_root", "", str))

        self.mat_path.setText("")
        self.preset.setCurrentIndex(0)

        self.m_basecolor.edit.setText("")
        self.m_illum.edit.setText("")
        self.m_ao.edit.setText("")
        self.m_normal.edit.setText("")
        self.m_bump.edit.setText("")
        self.m_metallic.edit.setText("")
        self.m_roughness.edit.setText("")
        self.m_height.edit.setText("")

        self.v_basecolor.edit.setText("")
        self.v_normal.edit.setText("")
        self.v_phong_src.edit.setText("")
        self.v_trans_mask.edit.setText("")
        self.v_emissive_tex.edit.setText("")

        self._apply_ui()

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()