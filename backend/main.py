# backend/main.py
import io
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Direct imports from our new sibling modules
from schemas import ChatRequest, ChatResponse, UploadResponse
from ai_logic import get_sql_with_rag, get_summary_from_ai, get_chart_type_from_ai
from core import get_db_engine

app = FastAPI(title="FloatChat Backend API")

origins = [
    "http://localhost:5173",
    "https://preview--ocean-whisperer-76.lovable.app",
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/upload", response_model=UploadResponse)
async def handle_file_upload(file: UploadFile = File(...)):
    if not file.filename.endswith('.nc'): raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        contents = await file.read()
        file_buffer = io.BytesIO(contents)
        with xr.open_dataset(file_buffer) as ds:
            wmo_id = ds['PLATFORM_NUMBER'].isel(N_PROF=0).values.item()
            project_name = ds.attrs.get('PROJECT_NAME', 'N/A').strip()
            start_date = pd.to_datetime(ds['JULD'].min().values).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(ds['JULD'].max().values).strftime('%Y-%m-%d')
            lat_range = f"{ds['LATITUDE'].min().values.item():.2f} to {ds['LATITUDE'].max().values.item():.2f}"
            lon_range = f"{ds['LONGITUDE'].min().values.item():.2f} to {ds['LONGITUDE'].max().values.item():.2f}"
            session_context_doc = f"""--- User Uploaded File Context ---
WMO ID: {int(wmo_id)}
Project Name: {project_name}
Data Collection Period: {start_date} to {end_date}
Geographic Area: Latitude [{lat_range}], Longitude [{lon_range}]"""
        return UploadResponse(filename=file.filename, message=f"Successfully processed {file.filename}. I am now aware of this float.", session_context=session_context_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

@app.post("/chat", response_model=ChatResponse)
async def handle_chat_message(request: ChatRequest):
    try:
        sql_query = get_sql_with_rag(request.message, request.session_context)
        engine = get_db_engine()
        with engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
        summary = get_summary_from_ai(request.message, result_df)
        chart_type = get_chart_type_from_ai(request.message, result_df.columns.to_list())
        data_for_frontend = result_df.to_dict(orient='records')
        return ChatResponse(summary=summary, sql_query=sql_query, chart_type=chart_type, data=data_for_frontend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))