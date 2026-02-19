#!/usr/bin/env python3
from flask import Flask, render_template, request, send_file, jsonify
import os
from pathlib import Path
import tempfile
import zipfile
import io
from process_svg_v2 import process_svg

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    files = request.files.getlist('files')
    params = {
        'outline_scale': float(request.form.get('outline_scale', 1.4)),
        'epsilon_factor': float(request.form.get('epsilon_factor', 0.00015)),
        'base_tension': float(request.form.get('base_tension', 0.6)),
        'angle_threshold': int(request.form.get('angle_threshold', 160)),
        'corner_tension_reduction': float(request.form.get('corner_tension_reduction', 0.0))
    }
    
    results = []
    
    for file in files:
        if not file.filename.endswith('.svg'):
            continue
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='wb') as tmp_in:
            file.save(tmp_in.name)
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
    
    return jsonify(results)

@app.route('/download-zip', methods=['POST'])
def download_zip():
    data = request.json
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for result in data:
            if result['success']:
                for file_info in result['files']:
                    zip_file.writestr(file_info['name'], file_info['content'])
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='processed_svgs.zip')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
