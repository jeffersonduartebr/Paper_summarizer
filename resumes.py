import os
import sys
import fitz          
import ollama        
import pytesseract   
from PIL import Image
import time
import torch
from typing import List, Optional
import logging


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

def log(text):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{timestamp} - {text}")


def get_ideal_chunk_size(default: int = 4000) -> int:
    """
    Calcula o tamanho ideal de chunk (em caracteres) baseado na memória da GPU.
    Se não houver GPU disponível, retorna o valor default.
    """
    if not torch.cuda.is_available():
        logger.warning(f"GPU não disponível. Usando chunk padrão de {default} chars.")
        return default

    try:
        props = torch.cuda.get_device_properties(0)
        total_mem_bytes = props.total_memory
        total_mem_gb = total_mem_bytes / (1024 ** 3)
        # Estimativa: 80k chars por GB de memória
        ideal = int(total_mem_gb * 80000)
        # Garante limites mínimo e máximo
        ideal = max(1024, min(ideal, default * 120))
        logger.info(f"GPU detectada: {total_mem_gb:.1f} GB - definindo chunk para {ideal} chars.")
        return ideal
    except Exception as e:
        logger.error(f"Falha ao obter propriedades da GPU: {e}")
        return default


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto de cada página. Se não houver texto nativo, aplica OCR.
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            full_text.append(text)
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang="eng+por").strip()
            if ocr_text:
                full_text.append(ocr_text)
            else:
                log(f"[Aviso] Página {page_num} sem texto em {os.path.basename(pdf_path)}.")

    return "\n\n".join(full_text)


def chunk_text(text: str, max_chars: Optional[int] = None) -> List[str]:
    """
    Divide o texto em pedaços de até max_chars caracteres,
    respeitando quebras de parágrafo.
    Se max_chars for None, usa get_ideal_chunk_size().
    """
    if max_chars is None:
        max_chars = get_ideal_chunk_size()

    chunks: List[str] = []
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


def summarize_chunk(chunk: str, model: str = "gemma3:27b") -> str:
    prompt = (
        "Você é um assistente que resume textos científicos. "
        "Forneça um resumo em português culto, valorizando as ideias principais do trecho abaixo:\n\n"
        f"{chunk}"
    )
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Resuma textos em português."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp["message"]["content"].strip()


def synthesize_summaries_single(summaries: list[str], model: str) -> str:
    """
    Sintetiza resumos de um único artigo em texto corrido.
    """
    joined = "\n\n".join(summaries)
    prompt = (
        "Você é um assistente especializado em resumos científicos. "
        "Com base nas ideias abaixo, produza um resumo coeso em texto corrido, "
        "em língua portuguesa culta, sintetizando as principais contribuições do artigo:\n\n"
        f"{joined}"
    )
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Síntese multi-resumos em texto corrido."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp["message"]["content"].strip()


def process_papers(directory: str, model: str = "gemma3:27b") -> dict[str, str]:
    """
    Lê todos os PDFs em 'directory', extrai texto, chunking e gera resumo para cada artigo.
    Retorna dicionário {título: resumo_texto}.
    """
    summaries = {}
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith('.pdf'):
            path = os.path.join(directory, fname)
            log(f"Processando {fname}...")
            text = extract_text_from_pdf(path)
            if not text:
                log(f"Nenhum texto extraído de {fname}.")
                continue
            chunks = list(chunk_text(text))
            log(f"{fname}: dividido em {len(chunks)} partes.")
            article_parts = []
            for i, chunk in enumerate(chunks, 1):
                log(f"{fname}: resumindo parte {i}/{len(chunks)}...")
                article_parts.append(summarize_chunk(chunk, model=model))
            summaries[fname.replace('.pdf','')] = synthesize_summaries_single(article_parts, model)
    return summaries


def synthesize_summaries(summaries: dict[str, str], model: str = "gemma3:27b") -> str:
    """
    Gera capítulo de discussão comparativa em texto corrido, sem formatação Markdown.
    """
    sections = []
    for title, text in summaries.items():
        sections.append(f"Artigo '{title}':\n{text}")
    combined = "\n\n".join(sections)

    prompt = (
        "Você é um especialista em educação e pesquisa científica. "
        "Com base nos resumos dos artigos abaixo, produza um capítulo de livro em texto corrido, "
        "em língua portuguesa conforme a norma culta da gramática brasileira. "
        "Compare os trabalhos destacando: \n"
        "- pontos comuns "
        "- conflitos teóricos ou metodológicos"
        "- contribuições principais de cada artigo "
        "- lacunas para futuras pesquisas \n"
        "Não use formatação Markdown. "
        "O texto deve ser detalhado, fluido e coeso, como um capítulo de livro. "
        "Não inclua referências bibliográficas. "
        "O texto deve ser claro e acessível, evitando jargões técnicos. "
        "O texto deve ser escrito em português culto.\n\n"
        "Resumos dos artigos:\n"
        f"\n{combined}"
    )
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Gere texto corrido comparativo sem usar Markdown."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp["message"]["content"].strip()


def main():
    os.makedirs("static", exist_ok=True)
    input_dir = "papers"
    output_file = os.path.join("static", "capitulo_comparativo.txt")

    log("Iniciando processamento de artigos...")
    article_summaries = process_papers(input_dir)
    if not article_summaries:
        log("Nenhum artigo processado. Abortando.")
        return

    log("Gerando capítulo comparativo em texto corrido...")
    chapter = synthesize_summaries(article_summaries)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(chapter)
    log(f"Capítulo salvo em: {output_file}")

if __name__ == "__main__":
    main()
