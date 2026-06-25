#!/bin/bash
set -e

echo "=== LatentLipSyncLite entrypoint ==="

# Verificar rclone
echo "[DEBUG] rclone version: $(rclone version | head -1)"

# 1) Generar rclone.conf desde variables de entorno (sin tocar el host)
echo "[DEBUG] Verificando credenciales R2..."
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
    echo "[OK] rclone configurado para R2."
    echo "[DEBUG] R2_BUCKET=${R2_BUCKET}"
    echo "[DEBUG] R2_ENDPOINT=${R2_ENDPOINT}"
else
    echo "WARN: credenciales R2 no presentes. Se asume contenido local en \$CHECKPOINTS_DIR."
fi

# 2) Sincronizar checkpoints y voces desde R2 al filesystem efimero
if [ -n "$R2_ACCESS_KEY_ID" ] && [ -n "$R2_BUCKET" ]; then
    mkdir -p "$CHECKPOINTS_DIR" "$VOCES_DIR"
    
    echo ""
    echo "=== Descargando checkpoints desde R2 ==="
    echo "[DEBUG] Bucket: r2:${R2_BUCKET}/checkpoints/"
    echo "[DEBUG] Destino: ${CHECKPOINTS_DIR}/"
    
    # Verificar contenido del bucket primero
    echo "[DEBUG] Contenido en R2:"
    rclone ls "r2:${R2_BUCKET}/checkpoints/" 2>&1 || echo "[ERROR] No se pudo listar checkpoints"
    
    # Calcular tamaño
    echo "[DEBUG] Tamaño total:"
    rclone size "r2:${R2_BUCKET}/checkpoints/" 2>&1 || true
    
    echo ""
    echo "--- Descargando (puede tardar varios minutos) ---"
    set -x
    rclone copy "r2:${R2_BUCKET}/checkpoints/" "$CHECKPOINTS_DIR/" \
        --transfers 8 --progress \
        2>&1 | tee /tmp/rclone_checkpoints.log
    RCLONE_CHECKPOINTS_EXIT=${PIPESTATUS[0]}
    set +x
    
    if [ $RCLONE_CHECKPOINTS_EXIT -eq 0 ]; then
        echo "[OK] Checkpoints descargados."
        echo "[DEBUG] Contenido local:"
        ls -lh "$CHECKPOINTS_DIR/" 2>&1 || echo "[ERROR] No se pudo listar directorio local"
    else
        echo "[ERROR] Fallo al descargar checkpoints (exit code: $RCLONE_CHECKPOINTS_EXIT)"
        echo "[DEBUG] Log de rclone:"
        cat /tmp/rclone_checkpoints.log
    fi
    
    echo ""
    echo "=== Descargando voces desde R2 ==="
    echo "[DEBUG] Bucket: r2:${R2_BUCKET}/voces/"
    echo "[DEBUG] Destino: ${VOCES_DIR}/"
    
    # Verificar contenido del bucket primero
    echo "[DEBUG] Contenido en R2:"
    rclone ls "r2:${R2_BUCKET}/voces/" 2>&1 || echo "[ERROR] No se pudo listar voces"
    
    echo ""
    echo "--- Descargando ---"
    set -x
    rclone copy "r2:${R2_BUCKET}/voces/" "$VOCES_DIR/" \
        --transfers 8 --progress \
        2>&1 | tee /tmp/rclone_voces.log
    RCLONE_VOCES_EXIT=${PIPESTATUS[0]}
    set +x
    
    if [ $RCLONE_VOCES_EXIT -eq 0 ]; then
        echo "[OK] Voces descargadas."
        echo "[DEBUG] Contenido local:"
        ls -lh "$VOCES_DIR/" 2>&1 || echo "[ERROR] No se pudo listar directorio local"
    else
        echo "[ERROR] Fallo al descargar voces (exit code: $RCLONE_VOCES_EXIT)"
        echo "[DEBUG] Log de rclone:"
        cat /tmp/rclone_voces.log
    fi
else
    echo "R2 no configurado; saltando descarga."
fi

echo ""
echo "=== Directorios de trabajo ==="
mkdir -p /app/temp /app/gradio_tmp

echo ""
echo "=== Lanzando app.py ==="
exec python app.py ${EXTRA_ARGS:-}
