import fitz  # PyMuPDF
import whisper
import os
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataIngestion:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.whisper_model = whisper.load_model("base")
        
        # Ensure the processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_all_data(self):
        """
        Recursively walks through Company/Quarter folders.
        Example Path: data/raw/NVIDIA/Q1-2025/report.pdf
        """
        all_chunks = []
        
        # 1. Recursively find all PDF and Audio files
        # rglob("**/*") looks into every subfolder
        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() not in ['.pdf', '.mp3', '.wav']:
                continue

            # 2. Extract Metadata from Folder Structure
            # file_path.parts might look like: ('data', 'raw', 'NVIDIA', 'Q1-2025', 'transcript.pdf')
            # We assume structure: data/raw/{Company}/{Quarter}/{file}
            parts = file_path.relative_to(self.data_dir).parts
            company = parts[0] if len(parts) > 1 else "Unknown"
            quarter = parts[1] if len(parts) > 2 else "FullYear"

            print(f"📂 Processing: {company} | {quarter} | {file_path.name}")

            # 3. Process the file based on type
            if file_path.suffix == '.pdf':
                text = self._extract_pdf(file_path)
                print("PDF text extraction complete.")
                file_type = "pdf"
            else:
                text = self._transcribe_audio(file_path)
                print("Audio transcription complete.")
                file_type = "audio"

            # 4. Chunk and tag with metadata
            file_chunks = self._chunk_text(text, file_path.name, company, quarter, file_type)
            all_chunks.extend(file_chunks)

            # 3. Save to Processed Folder
            self._save_to_processed(file_chunks, company, quarter, file_path.stem)

        return all_chunks
    def _save_to_processed(self, chunks, company, quarter, filename_stem):
        """Saves chunks to data/processed/{Company}/{Quarter}/{filename}.json"""
        target_dir = self.processed_dir / company / quarter
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / f"{filename_stem}.json"
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)
        print(f"💾 Saved processed chunks to: {target_file}")
    
    def _chunk_text(self, text, filename, company, quarter, file_type):
        """
        Smart splitting: Prevents mid-word cuts and maintains paragraph/sentence integrity.
        """
        # Initialize the 'Smart' Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Target size
            chunk_overlap=150,   # 'Context Bridge' - repeats the last 150 chars in the next chunk
            separators=["\n\n", "\n", " ", ""] # Try to split by Para, then Line, then Space
        )
        
        # Split the text smartly
        raw_chunks = text_splitter.split_text(text)
        
        # Wrap in your metadata format
        final_chunks = []
        for chunk in raw_chunks:
            final_chunks.append({
                "text": chunk,
                "source": filename,
                "company": company,
                "quarter": quarter,
                "type": file_type
            })
        return final_chunks

    def _extract_pdf(self, path):
        with fitz.open(path) as doc:
            return "".join([page.get_text() for page in doc])

    def _transcribe_audio(self, path):
        print(f"Loading Whisper model-'base'...")
        return self.whisper_model.transcribe(str(path))["text"]
    

if __name__ == "__main__":
    ingestor = DataIngestion()
    final_data = ingestor.process_all_data()
    print(f"✅ Extracted and Saved {len(final_data)} total chunks across all companies.")