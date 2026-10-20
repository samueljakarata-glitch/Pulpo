#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minidoc.py - Genera mini documentales animados verticales (9:16) a partir de un guion JSON.

Uso:
    python minidoc.py video guiones/pulpo.json            # con voz (necesita internet: edge-tts)
    python minidoc.py video guiones/pulpo.json --sin-voz  # sin voz (tiempos estimados)
    python minidoc.py video guiones/a.json guiones/b.json # varios videos (lote)

Salida: salida/<nombre-del-guion>.mp4

Dependencias: ffmpeg, pillow, numpy, opencv-python-headless (+ edge-tts para la voz).
"""
import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
LEAD = 0.4      # silencio antes de la primera frase (s)
GAP = 0.35      # pausa entre escenas (s)
TAIL = 0.9      # cola final tras la última palabra (s)
SUBS_Y = 1390   # centro vertical de los subtítulos
SUB_SIZE = 70   # tamaño de letra de los subtítulos

COLORES = {
    "blanco": (244, 246, 250), "turquesa": (46, 196, 182), "coral": (255, 107, 107),
    "dorado": (255, 209, 102), "azul": (86, 168, 255), "morado": (167, 139, 250),
    "gris": (150, 165, 195), "verde": (120, 220, 140),
}
PULPO = (226, 104, 142)
PULPO_CLARO = (255, 160, 185)
FONDO_ARRIBA = (11, 18, 40)
FONDO_ABAJO = (30, 12, 52)
TARJETA = (30, 42, 78)
OSCURO = (12, 20, 44)


def col(x, defecto="turquesa"):
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    return COLORES.get(x or defecto, COLORES[defecto])


# ----------------------------------------------------------------------------
# Utilidades matemáticas
# ----------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def e_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def e_back(x):
    x = clamp(x)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def e_io(x):
    x = clamp(x)
    return 3 * x * x - 2 * x ** 3


# ----------------------------------------------------------------------------
# Fuentes y texto (PIL)
# ----------------------------------------------------------------------------
RUTAS_FUENTE = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
_fuentes = {}


def fuente(size):
    size = int(size)
    if size not in _fuentes:
        for r in RUTAS_FUENTE:
            if os.path.exists(r):
                _fuentes[size] = ImageFont.truetype(r, size)
                break
        else:
            _fuentes[size] = ImageFont.load_default()
    return _fuentes[size]


def ancho(s, size):
    f = fuente(size)
    return max((f.getlength(l) for l in s.split("\n")), default=0)


def ajustar(s, size, maxw):
    while size > 28 and ancho(s, size) > maxw:
        size -= 3
    return size


class Cola:
    """Cola de textos: se dibujan todos juntos al final (una sola pasada PIL)."""

    def __init__(self):
        self.items = []

    def t(self, s, x, y, size, color=None, alpha=1.0, anchor="mm", borde=0,
          sombra=True, maxw=None, inter=1.18):
        if alpha <= 0.02 or not s:
            return
        if maxw:
            size = ajustar(s, size, maxw)
        self.items.append((s, x, y, int(size), color or COLORES["blanco"],
                           clamp(alpha), anchor, borde, sombra, inter))


def volcar(img, cola):
    if not cola.items:
        return img
    base = Image.fromarray(img).convert("RGBA")
    capa = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    for (s, x, y, size, color, alpha, anchor, borde, sombra, inter) in cola.items:
        f = fuente(size)
        lineas = s.split("\n")
        lh = size * inter
        y0 = y - (len(lineas) - 1) * lh / 2
        for i, l in enumerate(lineas):
            yy = y0 + i * lh
            if sombra:
                d.text((x, yy + 5), l, font=f, anchor=anchor,
                       fill=(0, 0, 0, int(120 * alpha)))
            kw = {}
            if borde:
                kw = dict(stroke_width=borde, stroke_fill=(0, 0, 0, int(255 * alpha)))
            d.text((x, yy), l, font=f, anchor=anchor,
                   fill=color + (int(255 * alpha),), **kw)
    return np.array(Image.alpha_composite(base, capa).convert("RGB"))


# ----------------------------------------------------------------------------
# Formas (OpenCV, con antialiasing)
# ----------------------------------------------------------------------------
def fondo():
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top = np.array(FONDO_ARRIBA, np.float32)[None, None, :]
    bot = np.array(FONDO_ABAJO, np.float32)[None, None, :]
    img = np.repeat(top * (1 - y) + bot * y, W, axis=1)
    img = np.ascontiguousarray(img.astype(np.uint8))
    for yy in range(60, H, 90):
        for xx in range(60, W, 90):
            cv2.circle(img, (xx, yy), 2, (38, 52, 92), -1, cv2.LINE_AA)
    return img


_brillos = {}


def brillo(img, cx, cy, r, color, fuerza=0.4):
    r = int(r)
    if fuerza <= 0.01 or r < 4:
        return
    if r not in _brillos:
        yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
        _brillos[r] = np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 2
    a = _brillos[r] * fuerza
    x0, y0 = int(cx) - r, int(cy) - r
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + 2 * r), min(H, y0 + 2 * r)
    if X1 <= X0 or Y1 <= Y0:
        return
    a = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
    roi = img[Y0:Y1, X0:X1].astype(np.float32)
    roi += (np.array(color, np.float32) - roi) * a
    img[Y0:Y1, X0:X1] = roi.astype(np.uint8)


def caja(img, x0, y0, x1, y1, r, color, alpha=1.0):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if alpha <= 0.01 or x1 <= x0 or y1 <= y0:
        return
    w, h = x1 - x0, y1 - y0
    r = int(max(0, min(r, w // 2, h // 2)))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)[..., None] / 255 * alpha
    roi = img[Y0:Y1, X0:X1].astype(np.float32)
    roi += (np.array(color, np.float32) - roi) * m
    img[Y0:Y1, X0:X1] = roi.astype(np.uint8)


def corazon(img, cx, cy, size, color):
    if size < 4:
        return
    ts = np.linspace(0, 2 * math.pi, 64)
    x = 16 * np.sin(ts) ** 3
    y = 13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts)
    k = size / 34.0
    pts = np.stack([cx + x * k, cy - (y + 2.5) * k], 1).astype(np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx - 7 * k), int(cy - 8 * k)), (int(3.2 * k), int(2 * k)),
                -30, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)


def gota(img, cx, cy, r, color):
    if r < 4:
        return
    R = 0.8 * r
    C = (cx, cy + 0.25 * r)
    P = (cx, cy - 1.15 * r)
    dd = C[1] - P[1]
    phi = math.acos(min(1.0, R / dd))
    arc = [(C[0] + R * math.sin(a), C[1] - R * math.cos(a))
           for a in np.linspace(phi, 2 * math.pi - phi, 40)]
    pts = np.array([P] + arc, np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx - 0.3 * r), int(cy + 0.25 * r)), (int(0.17 * r), int(0.3 * r)),
                20, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)


def cerebro(img, cx, cy, s, color, linea=OSCURO):
    if s < 6:
        return
    cx, cy = int(cx), int(cy)
    for sx in (-1, 1):
        cv2.circle(img, (int(cx + sx * 0.3 * s), cy), int(0.52 * s), color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, int(cy + 0.4 * s)), (int(0.22 * s), int(0.3 * s)), 0, 0, 360,
                color, -1, cv2.LINE_AA)
    th = max(2, int(0.045 * s))
    cv2.line(img, (cx, int(cy - 0.5 * s)), (cx, int(cy + 0.35 * s)), linea, th, cv2.LINE_AA)
    for sx in (-1, 1):
        cv2.ellipse(img, (int(cx + sx * 0.42 * s), int(cy - 0.12 * s)), (int(0.22 * s), int(0.16 * s)),
                    0, 190, 350, linea, th, cv2.LINE_AA)
        cv2.ellipse(img, (int(cx + sx * 0.38 * s), int(cy + 0.2 * s)), (int(0.2 * s), int(0.14 * s)),
                    0, 10, 170, linea, th, cv2.LINE_AA)


def mano(img, cx, cy, s, color):
    if s < 6:
        return
    caja(img, cx - 0.32 * s, cy - 0.02 * s, cx + 0.32 * s, cy + 0.55 * s, 0.14 * s, color)
    for i, off in enumerate((-0.225, -0.075, 0.075, 0.225)):
        h = (0.52, 0.62, 0.58, 0.46)[i] * s
        x = int(cx + off * s)
        th = int(0.13 * s)
        cv2.line(img, (x, int(cy + 0.05 * s)), (x, int(cy - h)), color, th, cv2.LINE_AA)
        cv2.circle(img, (x, int(cy - h)), th // 2, color, -1, cv2.LINE_AA)
    cv2.line(img, (int(cx - 0.3 * s), int(cy + 0.35 * s)), (int(cx - 0.52 * s), int(cy + 0.02 * s)),
             color, int(0.13 * s), cv2.LINE_AA)
    cv2.circle(img, (int(cx - 0.52 * s), int(cy + 0.02 * s)), int(0.065 * s), color, -1, cv2.LINE_AA)


def ojo(img, cx, cy, s, color):
    if s < 6:
        return
    cv2.ellipse(img, (int(cx), int(cy)), (int(0.62 * s), int(0.38 * s)), 0, 0, 360,
                (240, 244, 250), -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(0.3 * s), color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(0.14 * s), OSCURO, -1, cv2.LINE_AA)


def lupa(img, cx, cy, s, color):
    if s < 6:
        return
    cv2.circle(img, (int(cx - 0.1 * s), int(cy - 0.1 * s)), int(0.36 * s), color,
               max(3, int(0.12 * s)), cv2.LINE_AA)
    cv2.line(img, (int(cx + 0.16 * s), int(cy + 0.16 * s)), (int(cx + 0.5 * s), int(cy + 0.5 * s)),
             color, max(4, int(0.16 * s)), cv2.LINE_AA)


def rayo(img, cx, cy, s, color):
    if s < 6:
        return
    p = np.array([(0.12, -0.6), (-0.3, 0.08), (-0.02, 0.08), (-0.14, 0.62), (0.34, -0.1),
                  (0.04, -0.1)], np.float32) * s
    pts = (p + np.array([cx, cy], np.float32)).astype(np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


ICONOS = {"corazon": None, "gota": None, "cerebro": None, "mano": None, "ojo": None,
          "lupa": None, "rayo": None}


def icono(img, nombre, cx, cy, s, color):
    if nombre == "corazon":
        corazon(img, cx, cy, s, color)
    elif nombre == "gota":
        gota(img, cx, cy, 0.5 * s, color)
    elif nombre == "cerebro":
        cerebro(img, cx, cy, 0.8 * s, color)
    elif nombre == "mano":
        mano(img, cx, cy, s, color)
    elif nombre == "ojo":
        ojo(img, cx, cy, s, color)
    elif nombre == "lupa":
        lupa(img, cx, cy, s, color)
    elif nombre == "rayo":
        rayo(img, cx, cy, s, color)


def pulpo(img, cx, cy, s, t, color=PULPO):
    """Pulpo animado: tentáculos que ondulan, ojos que miran."""
    n = 8
    for i in range(n):
        u = (i - (n - 1) / 2) / ((n - 1) / 2)
        bx, by = cx + u * 0.42 * s, cy + 0.35 * s
        segs = 18
        L = 0.95 * s * (1 - 0.15 * abs(u))
        pts = []
        for k in range(segs + 1):
            q = k / segs
            sway = math.sin(t * 2.6 + i * 0.9 + q * 3.2) * 0.12 * s * q
            pts.append((bx + u * 0.30 * s * q + sway, by + q * L))
        for k in range(segs):
            th = int(max(4, 0.09 * s * (1 - k / segs)))
            p0 = (int(pts[k][0]), int(pts[k][1]))
            p1 = (int(pts[k + 1][0]), int(pts[k + 1][1]))
            cv2.line(img, p0, p1, color, th, cv2.LINE_AA)
            cv2.circle(img, p1, th // 2, color, -1, cv2.LINE_AA)
            if k % 3 == 1 and th > 8:
                cv2.circle(img, p0, max(2, th // 5), PULPO_CLARO, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx), int(cy - 0.08 * s)), (int(0.58 * s), int(0.66 * s)), 0, 0, 360,
                color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(cx - 0.2 * s), int(cy - 0.45 * s)), (int(0.16 * s), int(0.09 * s)),
                -30, 0, 360, PULPO_CLARO, -1, cv2.LINE_AA)
    mx, my = 0.025 * s * math.sin(t * 1.3), 0.015 * s * math.cos(t * 0.9)
    for sx in (-1, 1):
        ex, ey = int(cx + sx * 0.24 * s), int(cy + 0.02 * s)
        cv2.circle(img, (ex, ey), int(0.14 * s), (245, 247, 252), -1, cv2.LINE_AA)
        cv2.circle(img, (int(ex + mx), int(ey + my)), int(0.07 * s), OSCURO, -1, cv2.LINE_AA)
        cv2.circle(img, (int(ex + mx - 0.02 * s), int(ey + my - 0.025 * s)), max(2, int(0.02 * s)),
                   (255, 255, 255), -1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Escenas
# ----------------------------------------------------------------------------
def esc_gancho(img, q, e, t, d):
    a = e_out(t / 0.6)
    brillo(img, 540, 560, 520, col("morado"), 0.35 * a)
    tam = 260 * (0.85 + 0.15 * e_back(t / 0.7)) * (1 + 0.02 * math.sin(t * 4))
    pulpo(img, 540, 560 + 8 * math.sin(t * 2), tam, t)
    q.t(e.get("pregunta", "¿SABÍAS QUE…?"), 540, 275, 64, col("dorado"), e_out((t - 0.1) / 0.4))
    lineas = e.get("lineas", [])
    cols = ["dorado", "coral", "turquesa", "azul"]
    n = max(len(lineas), 1)
    paso = max(0.8, (d - 2.4) / n * 0.85)
    for i, l in enumerate(lineas):
        k = e_back((t - (0.9 + i * paso)) / 0.45)
        if k > 0:
            q.t(l, 540, 1010 + i * 105, 78 * (0.7 + 0.3 * clamp(k, 0, 1.2)), col(cols[i % 4]),
                clamp(k * 1.5), maxw=960)


def esc_numero(img, q, e, t, d):
    c = col(e.get("color", "turquesa"))
    if e.get("titulo"):
        q.t(e["titulo"].upper(), 540, 295, 56, col("gris"), e_out(t / 0.4))
    n = e.get("numero", 0)
    val = int(round(n * e_out((t - 0.2) / 1.1)))
    s = f"{val}{e.get('sufijo', '')}"
    pop = 0.75 + 0.25 * e_back((t - 0.15) / 0.6)
    brillo(img, 540, 620, 430, c, 0.22 * e_out(t / 0.6))
    q.t(s, 540, 620, 400 * pop, c, e_out(t / 0.3), maxw=900)
    q.t(e.get("unidad", ""), 540, 850, 72, col("blanco"), e_out((t - 0.5) / 0.5), maxw=940)
    ic = e.get("icono")
    tiene_ic = bool(ic) and 1 <= n <= 6
    if tiene_ic:
        sp = min(190, 880 / n)
        x0 = 540 - (n - 1) * sp / 2
        for i in range(int(n)):
            k = e_back((t - 0.9 - i * 0.25) / 0.45)
            if k > 0:
                lat = 1 + 0.07 * math.sin(t * 6 + i * 1.3)
                icono(img, ic, x0 + i * sp, 1000, 125 * k * lat, c if ic != "corazon" else col("coral"))
    yb = 1135 if tiene_ic else 1030
    for i, txt in enumerate(e.get("detalle", [])):
        k = e_out((t - 1.6 - i * 0.7) / 0.5)
        if k > 0:
            y = yb + i * 100
            caja(img, 90 + (1 - k) * 600, y - 42, 990 + (1 - k) * 600, y + 42, 42, TARJETA, 0.92)
            q.t(txt, 540 + (1 - k) * 600, y, 44, col("blanco"), clamp(k * 1.5), maxw=840)


def esc_gotas(img, q, e, t, d):
    q.t(e.get("titulo", "").upper(), 540, 295, 60, col("dorado"), e_out(t / 0.4), maxw=940)
    for lado, x, t0 in (("izquierda", 290, 0.6), ("derecha", 790, 1.5)):
        o = e[lado]
        c = col(o.get("color"))
        k = e_back((t - t0) / 0.6)
        if k > 0:
            brillo(img, x, 700, 300, c, 0.25 * clamp(k))
            gota(img, x, 700 + 6 * math.sin(t * 3 + x), 175 * k, c)
            q.t(o.get("etiqueta", "").upper(), x, 990, 62, c, clamp(k * 1.4), maxw=470)
            q.t(o.get("sub", ""), x, 1062, 44, col("blanco"), clamp(k * 1.4), maxw=470)
    q.t("VS", 540, 700, 72, col("gris"), e_out((t - 1.0) / 0.4))
    if e.get("nota"):
        k = e_out((t - (d * 0.55)) / 0.5)
        caja(img, 70, 1128, 1010, 1220, 46, TARJETA, 0.92 * k)
        q.t(e["nota"], 540, 1174, 46, col("blanco"), clamp(k * 1.5), maxw=880)


def esc_barras(img, q, e, t, d):
    q.t(e.get("titulo", ""), 540, 335, 76, col("blanco"), e_out(t / 0.4), maxw=940)
    if e.get("icono"):
        k = e_back((t - 0.2) / 0.6)
        icono(img, e["icono"], 540, 585, 150 * k * (1 + 0.04 * math.sin(t * 4)), col("dorado"))
    for i, p in enumerate(e["partes"]):
        y = 800 + i * 190
        k = e_out((t - (0.9 + i * 1.0)) / 1.0)
        c = col(p.get("color"))
        al = clamp(k * 2)
        q.t(p["etiqueta"], 90, y - 62, 48, col("blanco"), al, anchor="lm", maxw=700)
        caja(img, 90, y - 30, 990, y + 50, 40, TARJETA, 0.9 * al)
        wb = 900 * p["valor"] / 100 * k
        if wb > 40:
            caja(img, 90, y - 30, 90 + wb, y + 50, 40, c, 1.0)
        q.t(f"{int(round(p['valor'] * k))} %", 990, y - 62, 54, c, al, anchor="rm")
    if e.get("nota"):
        q.t(e["nota"], 540, 1180, 44, col("gris"), e_out((t - d * 0.6) / 0.5), maxw=900)


def esc_pasos(img, q, e, t, d):
    q.t(e.get("titulo", "").upper(), 540, 295, 56, col("gris"), e_out(t / 0.4), maxw=940)
    items = e["items"]
    n = len(items)
    cols = ["turquesa", "coral", "dorado", "azul"]
    for i, it in enumerate(items):
        t0 = 0.7 + i * max(0.8, (d - 2.0) / max(n, 1) * 0.8)
        k = e_out((t - t0) / 0.5)
        if k <= 0:
            continue
        y = 470 + i * 215
        dx = (1 - k) * 800
        c = col(it.get("color") or cols[i % 4])
        caja(img, 90 + dx, y - 85, 990 + dx, y + 85, 40, TARJETA, 0.93)
        cv2.circle(img, (int(190 + dx), y), 66, c, -1, cv2.LINE_AA)
        icono(img, it.get("icono", "rayo"), 190 + dx, y, 78, OSCURO)
        q.t(it["texto"], 290 + dx, y, 60, col("blanco"), clamp(k * 1.6), anchor="lm", maxw=670)


def esc_final(img, q, e, t, d):
    a = e_out(t / 0.6)
    brillo(img, 540, 560, 520, col("dorado"), 0.28 * a)
    pulpo(img, 540, 560 + 8 * math.sin(t * 2), 260 * (0.85 + 0.15 * e_back(t / 0.7)), t)
    frase = e.get("frase", "")
    n = len(frase.split("\n"))
    yf = 1000 + (n - 1) * 52
    q.t(frase, 540, yf, 88, col("dorado"), e_out((t - 0.5) / 0.5), maxw=960)
    if e.get("sub"):
        q.t(e["sub"], 540, yf + n * 52 + 70, 48, col("gris"), e_out((t - 1.2) / 0.5), maxw=900)


ESCENAS = {"gancho": esc_gancho, "numero": esc_numero, "gotas": esc_gotas,
           "barras": esc_barras, "pasos": esc_pasos, "final": esc_final}


# ----------------------------------------------------------------------------
# Tiempos: estimados o de la voz real. Mismo formato en ambos casos.
# ----------------------------------------------------------------------------
def estimar_escena(narracion):
    t, ev = 0.0, []
    for tok in narracion.split():
        d = 0.050 * len(tok) + 0.09
        ev.append((t, t + d, tok))
        t += d
        if tok[-1] in ".?!":
            t += 0.30
        elif tok[-1] in ",;:":
            t += 0.14
    return ev, t


def repartir(tokens, dur):
    pesos = [len(x) + 2 for x in tokens]
    k = dur * 0.96 / sum(pesos)
    t, ev = 0.0, []
    for tok, p in zip(tokens, pesos):
        ev.append((t, t + p * k, tok))
        t += p * k
    return ev


def armar_tiempos(eventos, durs):
    """eventos[i] = [(s, e, palabra)] relativos al inicio del audio de la escena i."""
    off, escenas, offsets = LEAD, [], []
    for i, (ev, dur) in enumerate(zip(eventos, durs)):
        offsets.append(off)
        ultimo = i == len(durs) - 1
        fin = off + dur + (TAIL if ultimo else GAP)
        escenas.append({
            "start": 0.0 if i == 0 else off, "end": fin,
            "words": [{"w": w, "s": off + s, "e": off + e} for (s, e, w) in ev]})
        off = fin
    return {"escenas": escenas, "offsets": offsets, "duracion": escenas[-1]["end"], "voz": None}


def estimar(guion):
    evs, durs = [], []
    for e in guion["escenas"]:
        ev, dur = estimar_escena(e["narracion"])
        evs.append(ev)
        durs.append(dur)
    return armar_tiempos(evs, durs)


# ----------------------------------------------------------------------------
# Voz (edge-tts) - necesita internet. Ensamblado con ffmpeg.
# ----------------------------------------------------------------------------
def duracion_audio(ruta):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(ruta)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


async def _sintetizar(texto, voz, velocidad, mp3):
    import edge_tts
    try:
        com = edge_tts.Communicate(texto, voz, rate=velocidad, boundary="WordBoundary")
    except TypeError:  # versiones antiguas: ya emiten WordBoundary por defecto
        com = edge_tts.Communicate(texto, voz, rate=velocidad)
    ev = []
    with open(mp3, "wb") as f:
        async for ch in com.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                s = ch["offset"] / 1e7
                ev.append((s, s + ch["duration"] / 1e7, ch["text"]))
    return ev


def ensamblar_voz(mp3s, offsets, destino, total):
    """Pone cada audio de escena en su instante exacto y los mezcla en un solo WAV."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for m in mp3s:
        cmd += ["-i", str(m)]
    partes = [f"[{i}:a]adelay={int(o * 1000)}:all=1[a{i}]" for i, o in enumerate(offsets)]
    mezcla = "".join(f"[a{i}]" for i in range(len(mp3s)))
    filtro = ";".join(partes) + f";{mezcla}amix=inputs={len(mp3s)}:normalize=0:dropout_transition=0,apad[o]"
    cmd += ["-filter_complex", filtro, "-map", "[o]", "-t", f"{total:.3f}", str(destino)]
    subprocess.run(cmd, check=True)


def hacer_voz(guion, trabajo):
    voz = guion.get("voz", "es-ES-AlvaroNeural")
    vel = guion.get("velocidad", "+0%")
    mp3s, durs, evs = [], [], []
    for i, e in enumerate(guion["escenas"]):
        mp3 = trabajo / f"escena{i + 1}.mp3"
        ev = None
        for intento in range(3):
            try:
                ev = asyncio.run(_sintetizar(e["narracion"], voz, vel, mp3))
                break
            except Exception as ex:  # red, servicio, versión
                print(f"  voz escena {i + 1}: intento {intento + 1} falló: {ex}")
                time.sleep(2)
        if ev is None:
            raise SystemExit("No se pudo generar la voz. Revisa internet o usa --sin-voz.")
        dur = duracion_audio(mp3)
        toks = e["narracion"].split()
        if len(ev) < 0.6 * len(toks):  # sin marcas por palabra: se reparten por longitud
            ev = repartir(toks, dur)
        mp3s.append(mp3)
        durs.append(dur)
        evs.append(ev)
    tim = armar_tiempos(evs, durs)
    wav = trabajo / "voz.wav"
    ensamblar_voz(mp3s, tim["offsets"], wav, tim["duracion"])
    tim["voz"] = str(wav)
    return tim


# ----------------------------------------------------------------------------
# Música ambiental generada por código (sin derechos de autor)
# ----------------------------------------------------------------------------
def generar_musica(destino, segundos, sr=44100):
    acordes = [(110.0, 220.0, 261.63, 329.63, 392.0), (87.31, 174.61, 220.0, 261.63, 329.63),
               (65.41, 196.0, 261.63, 329.63, 392.0), (98.0, 196.0, 246.94, 293.66, 329.63)]
    dur_a = 6.5
    n = int(sr * (segundos + dur_a))
    t = np.arange(n) / sr
    izq, der = np.zeros(n), np.zeros(n)
    i = 0
    while i * dur_a < segundos + dur_a:
        a = acordes[i % 4]
        ini = int(i * dur_a * sr)
        fin = min(n, int((i + 1) * dur_a * sr + dur_a * 0.6 * sr))
        tt = t[ini:fin]
        loc = np.arange(len(tt)) / sr
        env = np.clip(np.minimum(loc / 2.2, (loc[-1] - loc) / 2.2 if len(loc) else 1), 0, 1) ** 1.5
        s_i = np.zeros(len(tt))
        s_d = np.zeros(len(tt))
        for j, f in enumerate(a):
            amp = 0.5 if j == 0 else 0.22
            s_i += amp * np.sin(2 * np.pi * f * 1.002 * tt) + 0.08 * np.sin(4 * np.pi * f * tt)
            s_d += amp * np.sin(2 * np.pi * f * 0.998 * tt) + 0.08 * np.sin(4 * np.pi * f * tt)
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.18 * tt + i)
        izq[ini:fin] += s_i * env * lfo
        der[ini:fin] += s_d * env * lfo
        i += 1
    pico = max(np.abs(izq).max(), np.abs(der).max(), 1e-6)
    est = (np.stack([izq, der], 1) / pico * 0.6 * 32767).astype(np.int16)
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(est[: int(sr * segundos)].tobytes())


def preparar_audio(guion, tim, trabajo):
    D = tim["duracion"]
    musica = guion.get("musica", "auto")
    ruta_m = None
    if musica == "auto":
        ruta_m = trabajo / "musica.wav"
        generar_musica(ruta_m, D + 1)
    elif musica and musica != "ninguna" and os.path.exists(musica):
        ruta_m = musica
    voz = tim.get("voz")
    if not voz and not ruta_m:
        return None
    destino = trabajo / "audio.m4a"
    fade = f"afade=t=in:d=1.5,afade=t=out:st={max(D - 2.2, 0):.2f}:d=2.2"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if voz and ruta_m:
        cmd += ["-i", str(voz), "-i", str(ruta_m), "-filter_complex",
                f"[1:a]volume=0.13,{fade}[m];[0:a]apad[v];[v][m]amix=inputs=2:duration=longest:normalize=0[a]",
                "-map", "[a]"]
    elif voz:
        cmd += ["-i", str(voz), "-af", "apad", ]
    else:
        cmd += ["-i", str(ruta_m), "-af", f"volume=0.45,{fade}"]
    cmd += ["-t", f"{D:.3f}", "-c:a", "aac", "-b:a", "160k", str(destino)]
    subprocess.run(cmd, check=True)
    return destino


# ----------------------------------------------------------------------------
# Subtítulos sincronizados
# ----------------------------------------------------------------------------
def trozos(tim):
    """Agrupa palabras en frases cortas de subtítulo con su ventana de tiempo."""
    lista = []
    f = fuente(SUB_SIZE)
    corto = {"y", "un", "una", "a", "el", "la", "los", "las", "de", "del", "en", "su", "sus",
             "no", "le", "con", "por", "que", "es", "al", "o", "se", "lo", "me", "mi", "tu"}
    for es in tim["escenas"]:
        cur, grupos = [], []
        for w in es["words"]:
            # si al añadir la palabra no cabe en una línea, se cierra la frase antes
            if cur and f.getlength(" ".join(x["w"] for x in cur + [w])) > 940:
                resto = []
                if len(cur) > 1 and cur[-1]["w"].lower() in corto:
                    resto = [cur.pop()]
                grupos.append(cur)
                cur = resto
            cur.append(w)
            fin_frase = w["w"][-1:] in ".,;:?!" and len(cur) >= 2
            lleno = len(cur) >= 4 or f.getlength(" ".join(x["w"] for x in cur)) >= 760
            if fin_frase or lleno:
                resto = []
                if not fin_frase and len(cur) > 1 and cur[-1]["w"].lower() in corto:
                    resto = [cur.pop()]  # no dejar "y", "un", "a"... colgando al final
                grupos.append(cur)
                cur = resto
        if cur:
            unido = grupos[-1] + cur if grupos else None
            if (unido and len(cur) == 1 and len(grupos[-1]) < 4
                    and f.getlength(" ".join(x["w"] for x in unido)) <= 940):
                grupos[-1] = unido
            else:
                grupos.append(cur)
        for gi, g in enumerate(grupos):
            t1 = grupos[gi + 1][0]["s"] - 0.03 if gi + 1 < len(grupos) else min(es["end"], g[-1]["e"] + 0.45)
            lista.append({"t0": g[0]["s"] - 0.03, "t1": t1, "w": g})
    return lista


def subtitulos(img, q, lista, t):
    ch = None
    for c in lista:
        if c["t0"] <= t < c["t1"]:
            ch = c
            break
    if not ch:
        return
    size = SUB_SIZE
    f = fuente(size)
    palabras = [w["w"] for w in ch["w"]]
    esp = f.getlength(" ")
    anchos = [f.getlength(p) for p in palabras]
    total = sum(anchos) + esp * (len(palabras) - 1)
    lineas = [list(range(len(palabras)))]
    if total > 980 and len(palabras) > 1:  # solo en casos extremos (palabras larguísimas)
        mitad = max(1, len(palabras) // 2)
        lineas = [list(range(mitad)), list(range(mitad, len(palabras)))]
    activa = max([i for i, w in enumerate(ch["w"]) if w["s"] <= t] or [0])
    lh = 90
    y0 = SUBS_Y - (len(lineas) - 1) * lh / 2
    maxw = 0
    for li, idxs in enumerate(lineas):
        wl = sum(anchos[i] for i in idxs) + esp * (len(idxs) - 1)
        maxw = max(maxw, wl)
    h = len(lineas) * lh
    caja(img, 540 - maxw / 2 - 34, SUBS_Y - h / 2 - 6, 540 + maxw / 2 + 34, SUBS_Y + h / 2 + 6, 36,
         (4, 6, 16), 0.62)
    for li, idxs in enumerate(lineas):
        wl = sum(anchos[i] for i in idxs) + esp * (len(idxs) - 1)
        x = 540 - wl / 2
        y = y0 + li * lh
        for i in idxs:
            color = col("dorado") if i == activa else col("blanco")
            q.t(palabras[i], x, y, size, color, 1.0, anchor="lm", borde=3, sombra=False)
            x += anchos[i] + esp


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def render(guion, tim, audio, salida):
    escenas = guion["escenas"]
    for i, e in enumerate(escenas):
        if e.get("tipo") not in ESCENAS:
            raise SystemExit(f"Escena {i + 1}: tipo '{e.get('tipo')}' desconocido. Opciones: {', '.join(ESCENAS)}")
    D = tim["duracion"]
    N = int(round(D * FPS))
    starts = [s["start"] for s in tim["escenas"]]
    ends = [s["end"] for s in tim["escenas"]]
    lista = trozos(tim)
    bg = fondo()
    serie = guion.get("serie", "MINI DOCUMENTAL")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio)]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += [str(salida)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    si = 0
    t_ini = time.time()
    for f in range(N):
        t = f / FPS
        while si < len(escenas) - 1 and t >= ends[si]:
            si += 1
        e = escenas[si]
        ts = t - starts[si]
        d = ends[si] - starts[si]
        img = bg.copy()
        q = Cola()
        ESCENAS[e["tipo"]](img, q, e, ts, d)
        img = volcar(img, q)
        p = clamp(ts / d)
        z = 1 + 0.06 * e_io(p)
        M = cv2.getRotationMatrix2D((W / 2, H * 0.42), 0, z)
        M[0, 2] += (18 if si % 2 == 0 else -18) * p
        M[1, 2] += -10 * p
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        a = min(1.0, ts / 0.3)
        if si < len(escenas) - 1:
            a = min(a, clamp((d - ts) / 0.2))
        a = min(a, clamp((D - t) / 0.5))
        if a < 1.0:
            img = cv2.addWeighted(img, a, bg, 1 - a, 0)
        img = np.ascontiguousarray(img)
        # HUD: barra de progreso, etiqueta y subtítulos
        caja(img, 60, 40, 1020, 50, 5, (60, 74, 112), 0.8)
        caja(img, 60, 40, 60 + 960 * t / D, 50, 5, col("dorado"), 1.0)
        q2 = Cola()
        q2.t(serie, 540, 185, 32, col("gris"), 0.9)
        subtitulos(img, q2, lista, t)
        img = volcar(img, q2)
        proc.stdin.write(img.tobytes())
        if f % 150 == 0:
            print(f"  cuadro {f}/{N}  ({time.time() - t_ini:.0f}s)", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg falló al crear el MP4")
    print(f"  listo en {time.time() - t_ini:.0f}s -> {salida}")


# ----------------------------------------------------------------------------
# Línea de comandos
# ----------------------------------------------------------------------------
def validar(guion):
    if not guion.get("escenas"):
        raise SystemExit("El guion no tiene 'escenas'.")
    for i, e in enumerate(guion["escenas"]):
        if not e.get("narracion"):
            raise SystemExit(f"Escena {i + 1}: falta 'narracion'.")


def hacer_video(ruta, sin_voz, salida):
    guion = json.loads(Path(ruta).read_text(encoding="utf-8"))
    validar(guion)
    slug = Path(ruta).stem
    trabajo = Path(salida) / f"_{slug}"
    trabajo.mkdir(parents=True, exist_ok=True)
    palabras = sum(len(e["narracion"].split()) for e in guion["escenas"])
    print(f"[{slug}] {len(guion['escenas'])} escenas, {palabras} palabras")
    tim = estimar(guion) if sin_voz else hacer_voz(guion, trabajo)
    D = tim["duracion"]
    print(f"  duración {'estimada' if sin_voz else 'real'}: {D:.1f} s")
    if not 45 <= D <= 55:
        print("  AVISO: fuera del rango 45-55 s. Ajusta el largo de la narración "
              f"(aprox. {palabras} palabras ahora).")
    audio = preparar_audio(guion, tim, trabajo)
    destino = Path(salida) / f"{slug}.mp4"
    render(guion, tim, audio, destino)
    return destino


def main():
    ap = argparse.ArgumentParser(description="Mini documentales verticales desde un guion JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video", help="genera uno o varios videos")
    v.add_argument("guiones", nargs="+")
    v.add_argument("--sin-voz", action="store_true", help="no genera voz (tiempos estimados)")
    v.add_argument("--salida", default="salida")
    a = ap.parse_args()
    ok, fallos = [], []
    for g in a.guiones:
        try:
            ok.append(hacer_video(g, a.sin_voz, a.salida))
        except (SystemExit, Exception) as ex:  # un guion malo no detiene el lote
            print(f"[{g}] ERROR: {ex}")
            fallos.append(g)
    print(f"\nTerminados: {len(ok)}  Con error: {len(fallos)}")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
