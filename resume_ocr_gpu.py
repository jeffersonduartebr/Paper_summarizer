import os
import sys
import torch
import fitz              # PyMuPDF
import ollama            # pip install ollama
import easyocr           # pip install easyocr
import numpy as np       # para converter PIL Image em numpy array
import logging
from PIL import Image
from io import BytesIO
from typing import List

# ----------------------------
# Configuração de Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Inicialização do EasyOCR
# ----------------------------
try:
    reader = easyocr.Reader(['en', 'pt'], gpu=torch.cuda.is_available())
    logger.info("EasyOCR reader inicializado com sucesso.")
except Exception as e:
    logger.error(f"Falha ao inicializar EasyOCR: {e}")
    sys.exit(1)

# ----------------------------
# Funções principais
# ----------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto de cada página:
      1) texto nativo via PyMuPDF
      2) fallback OCR via EasyOCR
    Retorna todo o texto concatenado.
    """
    if not os.path.isfile(pdf_path):
        logger.error(f"Arquivo PDF não encontrado: {pdf_path}")
        return ""

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Erro ao abrir PDF: {e}")
        return ""

    full_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            logger.info(f"Página {page_num}: texto nativo extraído ({len(text)} chars).")
            full_text.append(text)
            continue

        # sem texto nativo → OCR
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # converte PIL Image em array numpy para EasyOCR
        img_array = np.array(img)
        # easyocr retorna lista de strings (quando detail=0)
        try:
            results = reader.readtext(
                img_array,
                detail=0,         # só o texto
                paragraph=True    # junta linhas em parágrafo
            )
        except Exception as e:
            logger.error(f"Erro no OCR da página {page_num}: {e}")
            results = []
        ocr_text = "\n".join(results).strip()
        if ocr_text:
            logger.info(f"Página {page_num}: texto via OCR extraído ({len(ocr_text)} chars).")
            full_text.append(ocr_text)
        else:
            logger.warning(f"Página {page_num}: sem texto encontrado nem via OCR.")

    return "\n\n".join(full_text)


def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    Divide o texto em pedaços de até max_chars,
    respeitando quebras de parágrafo.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + max_chars
        if end < length:
            split = text.rfind("\n\n", start, end)
            if split != -1:
                end = split
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def summarize_chunk(chunk: str, model: str = "gemma3:12b") -> str:
    """
    Envia um chunk ao Ollama e retorna o resumo em português (norma culta).
    """
    prompt = (
        "Você é um assistente útil. "
        "Por favor, forneça um resumo conciso em língua portuguesa, seguindo a norma culta da gramática brasileira, do seguinte texto:\n\n"
        f"{chunk}"
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Você resume textos em português."},
                {"role": "user",   "content": prompt}
            ]
        )
        summary = resp["message"]["content"].strip()
        logger.info(f"Chunk resumido ({len(summary)} chars).")
        return summary
    except Exception as e:
        logger.error(f"Erro ao resumir chunk: {e}")
        return ""


def synthesize_summaries(summaries: List[str], model: str = "gemma3:12b") -> str:
    """
    Agrupa resumos parciais em um único resumo final em português (norma culta) em Markdown.
    """
    joined = "\n\n".join(f"- {s}" for s in summaries if s)
    prompt = (
        "Você é um especialista em síntese. "
        "Com base nos resumos a seguir, produza um único resumo coeso e estruturado, "
        "em língua portuguesa conforme a norma culta da gramática brasileira, no formato Markdown:\n\n"
        f"{joined}"
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Você sintetiza múltiplos resumos em português."},
                {"role": "user",   "content": prompt}
            ]
        )
        final = resp["message"]["content"].strip()
        logger.info("Síntese final gerada.")
        return final
    except Exception as e:
        logger.error(f"Erro ao sintetizar resumos: {e}")
        return ""


def main(pdf_path: str, output_path: str, model: str = "gemma3:12b"):
    logger.info("Iniciando processamento do PDF.")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.error("Nenhum texto extraído; abortando.")
        return

    chunks = chunk_text(text, max_chars=4000)
    logger.info(f"Texto dividido em {len(chunks)} chunks.")

    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        logger.info(f"Resumindo chunk {idx}/{len(chunks)}…")
        summaries.append(summarize_chunk(chunk, model=model))

    logger.info("Sintetizando resumo final…")
    final_summary = synthesize_summaries(summaries, model=model)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        logger.info(f"Resumo final salvo em: {output_path}")
    except Exception as e:
        logger.error(f"Falha ao salvar arquivo: {e}")


if __name__ == "__main__":
    PDF_FILE = "almeida.pdf"
    OUT_FILE = "resumo_gpu.md"

    # Garante diretório estático para eventuais dependências
    os.makedirs("static", exist_ok=True)

    main(PDF_FILE, OUT_FILE)
