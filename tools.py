from __future__ import annotations
import wikipedia
from smolagents import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
import subprocess, tempfile, os, json, csv, io, math, datetime, textwrap, pathlib, requests, urllib.parse, shutil, uuid, subprocess, shutil, bs4
from typing import List, Dict, Any, Set
import re
import subprocess
from duckduckgo_search import DDGS
import pandas as pd
from faster_whisper import WhisperModel          # CPU / GPU friendly
from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )


@tool
def execute_python_file(file_path: str) -> str:
    """
    Executes a Python file and returns all the output.

    Args:
        file_path: Path to the .py file to be executed

    Returns:
        All output from the program (stdout + stderr)
    """
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=60  # timeout de 60 segundos
        )
        
        # Combinar stdout y stderr
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
            
        return output if output else "Archivo ejecutado sin salida"
        
    except subprocess.TimeoutExpired:
        return "Error: El script excedió el tiempo límite de 60 segundos"
    except FileNotFoundError:
        return f"Error: No se encontró el archivo '{file_path}'"
    except Exception as e:
        return f"Error al ejecutar el archivo: {str(e)}"

@tool
def search_web_links_only(query: str, max_results: int = 3) -> str:
    """
    Busca en la web usando DuckDuckGo y devuelve solo los enlaces.
    
    Args:
        query: Consulta de búsqueda
        max_results: Número máximo de resultados (default: 3)
    
    Returns:
        JSON string con título y URL de cada resultado
    """
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:4000]
                })
            
            return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error en la búsqueda: {str(e)}"

# A tiny English word list to decide which direction contains "more real words"
BASIC_WORDS: Set[str] = {
    "the", "and", "of", "to", "in", "is", "was", "you",
    "are", "for", "that", "have", "with", "not", "this",
    "but", "they", "his", "her", "she", "him", "it",
    "on", "at", "as", "be", "or", "we", "he", "an",
    "if", "their", "will", "what", "so", "all",
}

@tool
def reverse_if_reversed(text: str) -> str:
    """
    If you can't understande the content of the text, maybe it's reversed.
    With this tool you can return the corrected (left-to-right) string, otherwise return *text* unchanged.

    Args:
        text: The original text that may or may not be reversed.

    Returns:
        The corrected text when reversal is detected, or the original
        text when it already reads left-to-right.
    """
    stripped = text.strip()
    if len(stripped) < 4:                       # too short to matter
        return text

    rev = stripped[::-1]

    # crude tokenisation: letters only
    toks_fwd = re.findall(r"[A-Za-z]+", stripped.lower())
    toks_rev = re.findall(r"[A-Za-z]+", rev.lower())

    real_fwd = sum(tok in BASIC_WORDS for tok in toks_fwd)
    real_rev = sum(tok in BASIC_WORDS for tok in toks_rev)

    if real_rev >= 2 * max(1, real_fwd):        # at least twice as many words
        return rev
    return text

@tool
def download_file(task_id: str, save_as: str) -> str:
    """Download a file from a task_id and save it in ./output.

    Args:
        task_id:  ID of the current question
        save_as:  Desired filename (e.g. "sales.xlsx").
                  If None the name is taken from Content-Disposition or, failing that,
                  from the URL/task_id.

    Returns:
        Absolute path of the saved file when the download succeeds,
        or a short diagnostic string starting with "Error: …" if something
        goes wrong (network failure, unsupported scheme, etc.).
    """
    url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"

    # carpeta de destino
    out_dir = pathlib.Path("./output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # descarga
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return f"Error: {exc}"

    # nombre: prioridad → save_as > Content-Disposition > fallback
    if save_as:
        filename = save_as
    else:
        cd = r.headers.get("Content-Disposition", "")
        import re
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        filename = m.group(1) if m else f"{task_id}"

    dest = out_dir / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(r.raw, f)

    return str(dest)

@tool
def youtube_download(video_url: str,  audio_only: bool = True, fmt: str = "m4a") -> str:
    """Download a YouTube video (or just its audio) with yt-dlp.

    Args:
        video_url:  URL completa del vídeo.
        audio_only: True → descarga solo el audio; False → descarga vídeo+audio.
        fmt:        Contenedor de salida cuando audio_only=True (m4a, mp3…).

    Returns:
        Ruta absoluta del archivo descargado, o un mensaje de error.
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    try:
        if audio_only:
            out_path = os.path.join(tmpdir, f"audio.{fmt}")
            ytdlp_fmt = f"bestaudio[ext={fmt}]/bestaudio"
        else:
            out_path = os.path.join(tmpdir, "video.mp4")
            ytdlp_fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"

        subprocess.run(
            [
                "yt-dlp",
                "-f", ytdlp_fmt,
                "-o", out_path,
                video_url,
                "--quiet",
                "--no-warnings",
            ],
            check=True,
        )
        return out_path

    except FileNotFoundError:
        return "yt-dlp not installed."
    except subprocess.CalledProcessError as e:
        return f"yt-dlp failed: {e}"
    except Exception as e:
        return f"Download error: {e}"

@tool
def fetch_url(
    url: str,
    start_idx: int = 0
) -> str:
    """
    Download a web page, convert the HTML <body> to plain text,
    and return **only the slice** ``text[start_idx:start_idx+5000]``.
    If there is still information to analyze, a message will clarify it at the end of the search, so
    could be a good option to continue exploring the page until the required information is found.
    This tool is intended to be used after identifying the best option with search_web_links_only.

    Args:
        url : str
            Full http/https URL of the page to fetch.
        start_idx : int, optional 
            Starting index (inclusive) of the substring to return.
            Default **0** (beginning of the text).

    Slicing semantics
    -----------------
    After downloading the page, the tool:

    1. Parses the HTML and selects the entire ``<body>`` element.
    2. Flattens it to plain text using single-space separators.
    3. Applies ordinary Python slicing:

       ```python
       result = body_text[start_idx:start_idx+5000]
       ```

       *Example:* ``start_idx=1`` → characters 2-5002.

       Negative indices work exactly as in standard Python slices.

    Returns
        The requested text slice, or an ``"Error: …"`` string if the fetch fails (network error, HTTP 403/404, etc.).
    """
    step = 10000

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return f"Error: {exc}"

    soup = bs4.BeautifulSoup(resp.text, "html.parser")
    body_text = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
    total_len = len(body_text)
    end_idx = start_idx + step
    slice_text = body_text[start_idx:end_idx]
    next_start = end_idx if end_idx is not None else total_len
    remaining = max(total_len - next_start, 0)

    footer = (
        f"\n\nThere are still {remaining} characters in the web page that have "
        f"not been analyzed yet; you can continue using the start_idx of "
        f"{next_start}"
    )

    return slice_text + footer

@tool
def whisper_transcribe(audio_path: str, lang: str = "en") -> str:
    """Transcribe an audio file (wav / mp3 / m4a…) with Whisper.

    Args:
        audio_path: Path to the local audio file.
        lang: ISO-639-1 language code (default "en").

    Returns:
        The transcribed text (best segment-level approximation).
    """
    if WhisperModel is None:
        return "Whisper not installed."

    model = WhisperModel("base.en", device="auto")       # small, fits CPU/GPU
    segments, _ = model.transcribe(audio_path, language=lang)
    text = " ".join(seg.text.strip() for seg in segments)
    return text if text else "No speech detected."

@tool
def calculator(expr: str) -> str:
    """Evaluate a safe arithmetic expression.

    Supported operators: + - * / ** () and math functions (sqrt, sin…).

    Args:
        expr: Expression as plain text, e.g. "12 * (5 + 3) - 7".

    Returns:
        The result or an error message.
    """
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    try:
        res = eval(expr, {"__builtins__": {}}, allowed)
        return str(res)
    except Exception as e:
        return f"Error: {e}"

@tool
def csv_tool(path: str) -> Dict[str, Any]:
    """Load a CSV file and return basic info plus a preview.

    Args:
        path: Path to the local CSV file.

    Returns:
        A dict with 'shape', 'columns' and 'preview' keys, or an
        'error' key if pandas is missing.
    """
    if pd is None:
        return {"error": "pandas not installed."}

    df = pd.read_csv(path)
    preview = df.head(10).to_dict(orient="records")
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "preview": preview,
    }

@tool
def youtube_transcript_or_whisper(video_url: str, lang: str = "en") -> str:
    """Get a YouTube transcript. If none exists, fall back to Whisper-based transcription.

    Args:
        video_url: Full YouTube URL.
        lang: Preferred language code (default: "en").

    Returns:
        Plain-text transcript or an error message.
    """
    if YouTubeTranscriptApi is None:
        return "youtube_transcript_api not installed."

    video_id = video_url.split("v=")[-1].split("&")[0]

    # 1️⃣  Intenta descargar cualquier pista disponible
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=[lang, "en"]
        )
        return " ".join(seg["text"] for seg in transcript)

    except TranscriptsDisabled:
        # Captions totalmente deshabilitadas: caemos al plan B
        pass
    except Exception as e:
        # Otros errores (geoblock, etc.): seguimos al plan B
        pass

    # 2️⃣  Fallback a Whisper si está instalado
    if WhisperModel is None:
        return "No captions and faster-whisper not installed."

    try:
        tmpdir = tempfile.mkdtemp()
        audio_path = os.path.join(tmpdir, "audio.m4a")

        # Descargar sólo audio con yt_dlp (necesita estar instalado)
        subprocess.run(
            ["yt-dlp", "-f", "bestaudio", "-o", audio_path, video_url],
            check=True,
            capture_output=True,
        )

        model = WhisperModel("base.en", device="auto")
        segments, _ = model.transcribe(audio_path, language=lang)
        text = " ".join(seg.text.strip() for seg in segments)

        return text if text else "Whisper produced no speech."

    except FileNotFoundError:
        return "yt-dlp not installed; cannot download audio."
    except subprocess.CalledProcessError as e:
        return f"yt-dlp failed: {e.stderr.decode()[:300]}"
    except Exception as e:
        return f"Whisper transcription failed: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@tool
def excel_tool(path: str, sheet: str = "0") -> Dict[str, Any]:
    """Load an Excel sheet and return basic info and a preview.

    Args:
        path: Path to the XLS/XLSX file.
        sheet: Sheet index as string or sheet name (default "0").

    Returns:
        Dict with 'shape', 'columns', 'preview', or an 'error'.
    """
    if pd is None:
        return {"error": "pandas not installed."}

    try:
        # Try to convert to int first, if it fails use as string
        try:
            sheet_param = int(sheet)
        except ValueError:
            sheet_param = sheet
        
        df = pd.read_excel(path, sheet_name=sheet_param)
        preview = df.head(10).to_dict(orient="records")
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "preview": preview,
        }
    except Exception as e:
        return {"error": f"Failed to load Excel file: {str(e)}"}

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for information about an specific scientific paper, returning a maximum of 3 results.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "unknown")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

@tool
def wiki_search(query: str, lang: str = "en", start_idx: int = 0) -> str:
    """Return Wikipedia content.
       If there is still information to analyze, a message will clarify it at the end of the search, so
       could be a good option to continue exploring the page until the required information is found.

    Args:
        query: Topic or phrase to look up.
        lang: BCP‑47 language tag (default: "en").
        start_idx : int, optional 
            Starting index (inclusive) of the substring to return.
            Default **0** (beginning of the text).

    Returns:
        Wikipedia page content truncated to max_length characters,
        or a diagnostic message if the page was not retrieved.
    """

    wikipedia.set_lang(lang)
    max_length = 10000

    try:
        # Get the page object
        page = wikipedia.page(query, auto_suggest=True)
        
        # Extract the full content
        content = page.content
        
        # Truncate if it exceeds maximum length
        if len(content) > max_length:
            content = content[start_idx:max_length] + f"\n\n[... content truncated. Original length: {len(page.content)} characters]"
        
        # Include basic page information
        result = f"Title: {page.title}\n"
        result += f"URL: {page.url}\n\n"
        result += f"Content:\n{content}"
        
        remaining = len(page.content)
        next_start = start_idx + max_length
        footer = (
        f"\n\nThere are still {len(page.content)} characters in the web page that have "
        f"not been analyzed yet; you can continue using the start_idx of "
        f"{next_start}"
        )

        return result + footer

    except wikipedia.DisambiguationError as exc:
        # Topic is ambiguous – suggest a handful of alternatives
        opts: List[str] = exc.options[:5]
        return "Disambiguation – did you mean: " + ", ".join(opts) + "?"

    except wikipedia.PageError:
        return "No page matched your query."

    except Exception as exc:  # noqa: BLE001 – surface every other failure
        return f"Wikipedia lookup failed: {exc}"
