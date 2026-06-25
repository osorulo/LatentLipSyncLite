#!/bin/bash
set -e

echo "=== LatentLipSyncLite entrypoint ==="

# 1) Generar rclone.conf desde variables de entorno (sin tocar el host)
if [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_SECRET_ACCESS_KEY" ] && [ -n "$R2_ENDPOINT" ]; then
    mkdir -p /root/.config/rclone
    cat > /root/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = ${R2_ENDPOINT}
EOF
    echo "rclone configurado para R2."
else
    echo "WARN: credenciales R2 no presentes. Se asume contenido local en \$CHECKPOINTS_DIR."
fi

# 2) Sincronizar checkpoints y voces desde R2 al filesystem efimero
if [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_BUCKET" ]; then
    mkdir -p "$CHECKPOINTS_DIR" "$VOCES_DIR"
    echo "--- Descargando checkpoints desde R2 (puede tardar 2-5 min) ---"
    rclone copy "r2:${R2_BUCKET}/checkpoints/" "$CHECKPOINTS_DIR/" \
        --transfers 8 --progress \
        || echo "WARN: no se pudo descargar checkpoints."
    echo "--- Descargando voces desde R2 ---"
    rclone copy "r2:${R2_BUCKET}/voces/" "$VOCES_DIR/" \
        --transfers 8 --progress \
        || echo "WARN: no se pudo descargar voces."
else
    echo "R2 no configurado; saltando descarga."
fi

# 3) Directorios de trabajo
mkdir -p /app/temp /app/gradio_tmp

# 4) Lanzar la app (preserva cualquier flag extra que pase RunPod)
echo "=== Lanzando app.py ==="
exec python app.py ${EXTRA_ARGS:-}