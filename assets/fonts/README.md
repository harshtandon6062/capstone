# Bundled fonts

Noto Sans (Regular + Bold), SIL Open Font License 1.1.

Bundled rather than read from the system so the operator panel renders
identically on every machine. `ui_text.py` falls back to a system font, and
finally to Pillow's built-in bitmap font, if these files are missing.
