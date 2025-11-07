import whisper
import fitz  # PyMuPDF
import json
from pathlib import Path
import time
import glob

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts all text from a given PDF file."""
    text = ""
    print(f"📄 Extracting text from {pdf_path.name}...")
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        print("✅ PDF text extraction complete.")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def transcribe_audio(audio_path: Path, model_name: str = "base") -> dict:
    """Transcribes an audio file using Whisper and returns the result."""
    print(f"🎙️ Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"🔊 Transcribing {audio_path.name}... (This may take a while)")
    result = model.transcribe(str(audio_path), verbose=True)
    print("✅ Audio transcription complete.")
    return result

def process_pair(audio_file: Path, pdf_file: Path, output_dir: Path):
    """Process a single (audio, pdf) pair and save the result to JSON."""
    start_time = time.time()

    pdf_text = extract_text_from_pdf(pdf_file)
    transcription_result = transcribe_audio(audio_file)

    processed_data = {
        "source_audio": audio_file.name,
        "source_pdf": pdf_file.name,
        "pdf_text": pdf_text,
        "whisper_transcription": transcription_result
    }

    output_filename = f"{audio_file.stem}_processed.json"
    output_path = output_dir / output_filename

    print(f"💾 Saving processed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    print(f"✨ Finished {audio_file.name} in {end_time - start_time:.2f} seconds.")
    print(f"👉 Output: {output_path}\n")

def main():
    """Main function to process raw data files."""
    base_path = Path(__file__).resolve().parent.parent
    raw_data_path = base_path / "data" / "raw"
    processed_data_path = base_path / "data" / "processed"

    # ✅ Automatically find .mp3 and .pdf files, even in subfolders
    audio_files = glob.glob(str(raw_data_path / "**/*.mp3"), recursive=True)
    pdf_files = glob.glob(str(raw_data_path / "**/*.pdf"), recursive=True)

    if not audio_files:
        print("⚠️ No audio files found in data/raw/ or subdirectories.")
        return
    if not pdf_files:
        print("⚠️ No PDF files found in data/raw/ or subdirectories.")
        return

    # Take the first audio and PDF file found
    audio_file = Path(audio_files[0])
    pdf_file = Path(pdf_files[0])

    print(f"🎧 Found audio: {audio_file.name}")
    print(f"📄 Found PDF: {pdf_file.name}")

    processed_data_path.mkdir(exist_ok=True)

    start_time = time.time()

    # 1. Extract text from the official PDF transcript
    pdf_text = extract_text_from_pdf(pdf_file)

    # 2. Transcribe audio using Whisper
    transcription_result = transcribe_audio(audio_file)

    # 3. Structure and save the combined data
    processed_data = {
        "source_audio": audio_file.name,
        "source_pdf": pdf_file.name,
        "pdf_text": pdf_text,
        "whisper_transcription": transcription_result
    }

    output_filename = f"{audio_file.stem}_processed.json"
    output_path = processed_data_path / output_filename

    print(f"💾 Saving processed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    print(f"✨ Processing complete in {end_time - start_time:.2f} seconds.")
    print(f"👉 Your processed file is ready at: {output_path}")

if __name__ == "__main__":
    main()
