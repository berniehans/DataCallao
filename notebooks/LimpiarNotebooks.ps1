# Buscamos todos los archivos .ipynb en el directorio actual
$notebooks = Get-ChildItem -Path . -Filter "*.ipynb"

if ($notebooks.Count -eq 0) {
    Write-Host "No se encontraron archivos .ipynb en esta carpeta." -ForegroundColor Yellow
    exit
}

foreach ($file in $notebooks) {
    Write-Host "Revisando: $($file.Name)..." -NoNewline

    try {
        # Leemos el contenido del archivo tal cual (Raw)
        $content = Get-Content -Path $file.FullName -Raw -ErrorAction Stop
        $json = $content | ConvertFrom-Json

        # Verificamos si existe 'metadata' y dentro de ella 'widgets'
        if ($json.metadata -and $json.metadata.PSObject.Properties["widgets"]) {
            
            # Eliminamos la propiedad 'widgets'
            $json.metadata.PSObject.Properties.Remove("widgets")

            # Guardamos el archivo sobrescribiendo el anterior
            # IMPORTANTE: -Depth 100 es vital para que PowerShell no corte la estructura profunda del JSON
            $json | ConvertTo-Json -Depth 100 | Set-Content -Path $file.FullName -Encoding UTF8

            Write-Host " [LIMPIADO Y GUARDADO]" -ForegroundColor Green
        } else {
            Write-Host " [ESTABA LIMPIO]" -ForegroundColor Cyan
        }
    }
    catch {
        Write-Host " [ERROR]" -ForegroundColor Red
        Write-Host "  Detalle: $_" -ForegroundColor Gray
    }
}

Write-Host "`nListo. Ya puedes hacer git add/commit." -ForegroundColor Magenta