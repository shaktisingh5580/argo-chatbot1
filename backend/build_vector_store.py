# backend/build_vector_store.py
import os
import glob
import xarray as xr
import pandas as pd

# Use the new, recommended packages for LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def process_metadata(file_path: str) -> dict:
    """
    Extracts key metadata from a single NetCDF file.
    Returns a dictionary with the metadata or None if an error occurs.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # Extracting relevant information using .values.item() for safety
            wmo_id = ds['PLATFORM_NUMBER'].isel(N_PROF=0).values.item()
            project_name = ds.attrs.get('PROJECT_NAME', 'N/A').strip()
            
            # Find the range of dates and format them
            start_date = pd.to_datetime(ds['JULD'].min().values).strftime('%Y-%-m-%-d')
            end_date = pd.to_datetime(ds['JULD'].max().values).strftime('%Y-%-m-%-d')
            
            # Find the geographic boundaries and format them
            min_lat = ds['LATITUDE'].min().values.item()
            max_lat = ds['LATITUDE'].max().values.item()
            min_lon = ds['LONGITUDE'].min().values.item()
            max_lon = ds['LONGITUDE'].max().values.item()
            
            return {
                "wmo_id": int(wmo_id),
                "project_name": project_name,
                "date_range": f"{start_date} to {end_date}",
                "latitude_range": f"{min_lat:.2f} to {max_lat:.2f}",
                "longitude_range": f"{min_lon:.2f} to {max_lon:.2f}"
            }
    except Exception as e:
        # Print a helpful error message but allow the script to continue with other files
        print(f"--> WARNING: Skipping file {os.path.basename(file_path)} due to an error: {e}")
        return None

def main():
    """
    Scans the 'data' directory for NetCDF files, processes them, and adds their
    metadata to a persistent Chroma vector store in the 'chroma_db' directory.
    """
    data_folder = 'data'
    persist_directory = 'chroma_db'
    
    # --- Input Validation ---
    if not os.path.isdir(data_folder):
        print(f"--> ERROR: Data folder '{data_folder}' not found.")
        print("Please create a 'data' folder inside your 'backend' directory and add your NetCDF (.nc) files to it.")
        return

    netcdf_files = glob.glob(os.path.join(data_folder, '*.nc'))
    if not netcdf_files:
        print(f"--> WARNING: No NetCDF files (.nc) found in the '{data_folder}' directory. The AI's memory will be empty.")
        return
        
    print(f"Found {len(netcdf_files)} NetCDF files to process.")
    
    all_documents = []
    all_metadatas = []

    # --- Data Processing Loop ---
    for file_path in netcdf_files:
        print(f"Processing: {os.path.basename(file_path)}...")
        metadata = process_metadata(file_path)
        
        if metadata:
            # This is the text "document" that the AI will use as its memory for each float
            float_document = f"""--- ARGO Float Information Document ---
WMO ID: {metadata['wmo_id']}
Project Name: {metadata['project_name']}
Data Collection Period: {metadata['date_range']}
Geographic Area of Operation: Latitude Range [{metadata['latitude_range']}], Longitude Range [{metadata['longitude_range']}]
Available Parameters: Temperature, Salinity, Pressure."""
            
            all_documents.append(float_document)
            all_metadatas.append({"source": os.path.basename(file_path), "wmo_id": metadata["wmo_id"]})

    if not all_documents:
        print("--> ERROR: Could not generate any valid documents from the files. Aborting vector store creation.")
        return
        
    # --- Vector Store Creation ---
    print(f"\nInitializing embedding model and vector store at '{persist_directory}'...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        collection_name="argo_float_metadata",
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    
    print(f"Adding {len(all_documents)} new documents to the vector store...")
    vector_store.add_texts(texts=all_documents, metadatas=all_metadatas)
    
    print(f"\n✅ Successfully built the vector store.")
    print(f"The 'chroma_db' directory is now ready for the application.")

if __name__ == "__main__":
    main()
