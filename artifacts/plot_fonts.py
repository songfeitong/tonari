"""Load the locally installed Geist font without bundling font binaries."""

from matplotlib import font_manager


def load_geist() -> None:
    for path in font_manager.findSystemFonts():
        if "geist" in path.lower():
            font_manager.fontManager.addfont(path)
    for weight in ("normal", "bold"):
        font_manager.findfont(
            font_manager.FontProperties(family="Geist", weight=weight),
            fallback_to_default=False,
        )
