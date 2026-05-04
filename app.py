#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import os
from pathlib import Path
import tempfile
import zipfile
import io
from process_svg_v2 import process_svg

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/process")
async def process(
    files: List[UploadFile] = File(...),
    outline_scale: float = Form(1.4),
    epsilon_factor: float = Form(0.00015),
    base_tension: float = Form(0.6),
    angle_threshold: int = Form(160),
    corner_tension_reduction: float = Form(0.0)
):
    params = {
        'outline_scale': outline_scale,
        'epsilon_factor': epsilon_factor,
        'base_tension': base_tension,
        'angle_threshold': angle_threshold,
        'corner_tension_reduction': corner_tension_reduction
    }

    results = []

    for file in files:
        if not file.filename.endswith('.svg'):
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='wb') as tmp_in:
            tmp_in.write(await file.read())
            tmp_path = tmp_in.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = process_svg(
                    tmp_path, temp_dir,
                    params['outline_scale'], params['angle_threshold'],
                    params['corner_tension_reduction'], params['epsilon_factor'],
                    params['base_tension']
                )

                if result:
                    files_data = []
                    for path in result:
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                files_data.append({
                                    'name': Path(path).name,
                                    'content': f.read()
                                })

                    results.append({ 
                        'name': file.filename,
                        'success': True,
                        'files': files_data
                    })
                else:
                    results.append({'name': file.filename, 'success': False, 'error': 'Processing failed'})
        except Exception as e:
            results.append({'name': file.filename, 'success': False, 'error': str(e)})
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content=results)

class DownloadRequest(list):
    pass

from pydantic import BaseModel

class FileInfo(BaseModel):
    name: str
    content: str

class ProcessResult(BaseModel):
    success: bool
    files: Optional[List[FileInfo]] = []
    name: Optional[str] = None
    error: Optional[str] = None

@app.post("/download-zip")
async def download_zip(data: List[ProcessResult]):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for result in data:
            if result.success:
                for file_info in result.files:
                    zip_file.writestr(file_info.name, file_info.content)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={"Content-Disposition": "attachment; filename=processed_svgs.zip"}
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
