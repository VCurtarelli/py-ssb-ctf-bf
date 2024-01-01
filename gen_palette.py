import numpy as np
from colorsys import hsv_to_rgb


def rgb_to_html(RGB):
    # Converts RGB to HTML
    HTMLs = ['{:02X}'.format(v) for v in RGB]
    HTML = ''.join(HTMLs)
    return HTML


def gen_palette(S, V, ncolors, hshift=0):
    # Generates palette of colors
    letters = 'ACEBDF'
    hs = np.arange(ncolors)/ncolors
    colors = []
    for h in hs:
        rgb = hsv_to_rgb(h+hshift/360, S/100, V/100)
        RGB = tuple(round(i * 255) for i in rgb)
        HTML = rgb_to_html(RGB)
        colors.append(HTML)
    ncolors = [r'\definecolor{Light' + letters[i] + r'}{HTML}{' + colors[i] + r'}' for i in range(len(colors))]
    return ncolors