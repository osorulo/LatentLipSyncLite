from pathlib import Path

def validate_file(path, extensions, max_mb=500):
    path = Path(path)

    if not path.exists():
        raise ValueError("Archivo no existe")

    if path.suffix.lower() not in extensions:
        raise ValueError(f"Formato inválido: {path.suffix}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Archivo demasiado grande ({size_mb:.1f} MB)")
