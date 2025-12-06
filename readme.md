# Proyecto: app

Breve descripción
- Repositorio con varias versiones de la aplicación; el archivo funcional actual (entrypoint) es `app_v9.py`.
- Código orientado a procesamiento/modelado (archivos `model_architecture*.py`, `models/`, etc.).

Cómo ejecutar

1. Asegúrate de tener Python 3.8+ instalado.
2. Crea/activa un entorno virtual (recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # si existe
```

3. Ejecuta la aplicación principal:

```bash
python app_v9.py
```

Archivos importantes
- `app_v9.py`: archivo funcional / entrypoint actual.
- `app.py`, `app_v*.py`: versiones antiguas o experimentales.
- `model_architecture.py`, `model_architecture_v2.py`: definiciones de modelo.
- `models/`: carpeta para modelos guardados (puede contener archivos grandes — está ignorada por git si aplica).

Datos y caché
- Las carpetas de datos (`/data`, `/datasets`, etc.) y cachés (`.cache/`, `__pycache__/`) están excluidas en `.gitignore`.
- Coloca datasets fuera del control de versiones o en una ruta configurada por el código.

Contribuir
- Para cambios mayores, crea una rama, añade tests (si procede) y abre un PR.

Contacto
- Si necesitas ayuda con esta versión, comenta en el repositorio o contacta al responsable del proyecto.

Licencia
- Añade información de licencia aquí si aplica.

