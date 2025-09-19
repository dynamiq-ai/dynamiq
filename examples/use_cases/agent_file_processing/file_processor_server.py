#!/usr/bin/env python3
"""
Simplified FastAPI File Processor Server

A simple FastAPI server that accepts file uploads and returns file type and content.

To run this FastAPI server, use the following command:
python file_processor_server.py
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from examples.use_cases.agent_file_processing.agent_file_api import PORT

app = FastAPI(title="File Processor API", version="1.0.0")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'message': 'FastAPI File Processor Server is running'
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    return JSONResponse(content={"filename": file.filename, "message": f"File received successfully {contents}"})


if __name__ == '__main__':
    import uvicorn
    print("Starting FastAPI File Processor Server...")
    print(f"Server will be available at: http://localhost:{PORT}")
    print(f"API Documentation: http://localhost:{PORT}/docs")
    print("\nEndpoints:")
    print("  GET  / - Health check")
    print("  POST /upload - Upload file and get type/content")
    print("\nTo run this FastAPI server, use the following command:")
    print("python file_processor_server.py")
    print()

    uvicorn.run(app, host="127.0.0.1", port=PORT)
