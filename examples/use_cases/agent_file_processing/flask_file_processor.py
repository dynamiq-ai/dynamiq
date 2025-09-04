#!/usr/bin/env python3
"""
Flask File Processor Server

A simple Flask server that accepts file uploads and returns:
- File type (MIME type and extension)
- File content (text content for text files, base64 for binary files)
- File metadata (size, name, etc.)

Usage:
    python flask_file_processor.py

Endpoints:
    POST /upload - Upload a file and get its type and content
    GET / - Health check endpoint
"""

import base64
import io
import mimetypes
import os
from typing import Dict, Any

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'json', 'xml', 'html', 'md',
    'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'zip', 'tar', 'gz'
}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_type_info(filename: str, content: bytes) -> Dict[str, Any]:
    """Get comprehensive file type information."""
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = 'application/octet-stream'
    
    # Get file extension
    file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    # Determine if it's a text file
    text_types = {
        'text/', 'application/json', 'application/xml', 'application/javascript',
        'application/csv', 'application/x-yaml', 'application/x-python'
    }
    is_text_file = any(mime_type.startswith(prefix) for prefix in text_types)
    
    return {
        'mime_type': mime_type,
        'file_extension': file_extension,
        'is_text_file': is_text_file,
        'is_binary': not is_text_file
    }


def extract_file_content(content: bytes, file_type_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract content from file based on its type."""
    result = {
        'content_type': 'text' if file_type_info['is_text_file'] else 'binary',
        'content': None,
        'encoding': 'utf-8'
    }
    
    if file_type_info['is_text_file']:
        try:
            # Try to decode as UTF-8
            text_content = content.decode('utf-8')
            result['content'] = text_content
            result['encoding'] = 'utf-8'
        except UnicodeDecodeError:
            try:
                # Try other common encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text_content = content.decode(encoding)
                        result['content'] = text_content
                        result['encoding'] = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all text decodings fail, treat as binary
                    result['content_type'] = 'binary'
                    result['content'] = base64.b64encode(content).decode('ascii')
                    result['encoding'] = 'base64'
            except Exception:
                result['content_type'] = 'binary'
                result['content'] = base64.b64encode(content).decode('ascii')
                result['encoding'] = 'base64'
    else:
        # Binary file - encode as base64
        result['content'] = base64.b64encode(content).decode('ascii')
        result['encoding'] = 'base64'
    
    return result


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Flask File Processor Server is running',
        'max_file_size': f"{MAX_CONTENT_LENGTH / (1024*1024):.1f}MB",
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload file and return its type and content."""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please include a file in the "file" field'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'File type not allowed',
                'message': f'Allowed extensions: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read file content
        file_content = file.read()
        filename = secure_filename(file.filename)
        
        # Get file type information
        file_type_info = get_file_type_info(filename, file_content)
        
        # Extract content based on file type
        content_info = extract_file_content(file_content, file_type_info)
        
        # Prepare response
        response = {
            'success': True,
            'file_info': {
                'filename': filename,
                'original_filename': file.filename,
                'size_bytes': len(file_content),
                'size_mb': round(len(file_content) / (1024 * 1024), 2)
            },
            'file_type': file_type_info,
            'content': content_info,
            'metadata': {
                'upload_timestamp': request.environ.get('HTTP_DATE', 'unknown'),
                'user_agent': request.headers.get('User-Agent', 'unknown'),
                'content_length': request.content_length
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'message': str(e),
            'type': type(e).__name__
        }), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'message': f'File size exceeds maximum allowed size of {MAX_CONTENT_LENGTH / (1024*1024):.1f}MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Available endpoints: GET /, POST /upload'
    }), 404


if __name__ == '__main__':
    print("Starting Flask File Processor Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")
    print(f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    print("\nEndpoints:")
    print("  GET  / - Health check")
    print("  POST /upload - Upload file and get type/content")
    print("\nServer starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
