import re

import pdfplumber
from tqdm import tqdm

from config.settings import EMBEDDING_MODEL_NAME, HF_TOKEN, MAX_NEW_TOKENS
from ingestion.chunking import split_by_tokens
from utils.hf_loader import load_auto_tokenizer


def _resolve_page_indices(pages, total_pages):
    """
    将 pages 配置解析为去重排序后的 0-based 页面索引列表。
    支持以下格式（页码均为 1-based）：
      None                   -> 全部页面
      int N                  -> 第 N 页到末尾
      [start, end]           -> 单个区间（含两端）
      [[s1,e1],[s2,e2],...]  -> 多个区间（含两端）
    """
    if pages is None:
        return list(range(total_pages))
    if isinstance(pages, int):
        ranges = [(pages, total_pages)]
    elif isinstance(pages, list) and len(pages) > 0:
        if isinstance(pages[0], list):
            ranges = [(r[0], r[1]) for r in pages]
        elif isinstance(pages[0], int) and len(pages) == 2:
            ranges = [(pages[0], pages[1])]
        else:
            raise ValueError("pages 格式不合法，应为整数、[start,end] 或 [[s1,e1],[s2,e2],...] 形式")
    else:
        raise ValueError("pages 格式不合法，应为整数、[start,end] 或 [[s1,e1],[s2,e2],...] 形式")

    indices = set()
    for start, end in ranges:
        s = max(0, start - 1)
        e = min(total_pages, end)
        indices.update(range(s, e))
    return sorted(indices)


def extract_text_from_pdf(pdf_path, pages):
    """
    提取 PDF 指定页的文本，合并为一个字符串。
    pages 支持：None（全部）、int、[start,end]、[[s1,e1],[s2,e2],...]
    """
    text = ""
    print(f"正在提取文件{pdf_path}的文本...")
    with pdfplumber.open(pdf_path) as pdf:
        page_indices = _resolve_page_indices(pages, len(pdf.pages))
        for idx in tqdm(page_indices, desc=f"提取 {pdf_path} 文本进度", unit="页"):
            page_text = pdf.pages[idx].extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def apply_rules(text, rules):
    """根据规则清洗文本。"""
    # 预清洗：统一常见中英文引号，避免后续分词或正则阶段出现字符不一致。
    quote_translation = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "„": '"',
            "‟": '"',
            "＂": '"',
            "«": '"',
            "»": '"',
            "「": '"',
            "」": '"',
            "『": '"',
            "』": '"',
            "‘": "'",
            "’": "'",
            "‚": "'",
            "‛": "'",
            "＇": "'",
            "‹": "'",
            "›": "'",
        }
    )
    text = text.translate(quote_translation)

    if "header_pattern" in rules and rules["header_pattern"]:
        text = re.sub(rules["header_pattern"], "", text, flags=re.MULTILINE)
    if "footer_pattern" in rules and rules["footer_pattern"]:
        text = re.sub(rules["footer_pattern"], "", text, flags=re.MULTILINE)
    if "page_number_pattern" in rules and rules["page_number_pattern"]:
        text = re.sub(rules["page_number_pattern"], "", text, flags=re.MULTILINE)

    for extra in rules.get("extra_patterns", []):
        pattern = extra["pattern"]
        repl = extra.get("replacement", "")
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    if rules.get("remove_special_chars", False):
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s\.,;:!?()《》【】“”‘’\-]", "", text)

    text = re.sub(r"\n\s*\n", "\n", text)

    if rules.get("merge_line_breaks", False):
        lines = text.split("\n")
        merged = []
        buffer = []
        for line in lines:
            stripped = line.strip()
            if stripped == "":
                if buffer:
                    merged.append("".join(buffer))
                    buffer = []
            else:
                buffer.append(stripped)
        if buffer:
            merged.append("".join(buffer))
        text = "\n".join(merged)

    return text.strip()


_VALID_TABLE_SETTING_KEYS = {
    "vertical_strategy",
    "horizontal_strategy",
    "snap_tolerance",
    "intersection_tolerance",
    "explicit_vertical_lines",
    "explicit_horizontal_lines",
    "edge_min_length",
    "min_words_vertical",
    "min_words_horizontal",
    "keep_blank_chars",
    "text_tolerance",
    "text_x_tolerance",
    "text_y_tolerance",
}


def table_to_markdown(table, caption=None):
    """将二维列表表格转换为 Markdown。"""
    if not table or len(table) == 0:
        return ""

    cleaned = [
        [re.sub(r"\s+", " ", cell).strip() if cell is not None else "" for cell in row]
        for row in table
    ]

    rows_md = []
    if caption:
        rows_md.append(f"【{caption}】")

    header = cleaned[0]
    rows_md.append("| " + " | ".join(header) + " |")
    rows_md.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in cleaned[1:]:
        row_padded = row + [""] * (len(header) - len(row))
        rows_md.append("| " + " | ".join(row_padded[: len(header)]) + " |")

    return "\n".join(rows_md)


def extract_tables_from_pdf(pdf_path, pages, table_settings=None):
    """
    提取 PDF 中的表格，返回 Markdown 文本块列表。
    """
    if table_settings is None:
        table_settings = {}

    filtered_settings = {k: v for k, v in table_settings.items() if k in _VALID_TABLE_SETTING_KEYS}

    table_chunks = []
    print(f"正在提取文件 {pdf_path} 中的表格...")
    with pdfplumber.open(pdf_path) as pdf:
        page_indices = _resolve_page_indices(pages, len(pdf.pages))
        for idx in tqdm(page_indices, desc=f"提取 {pdf_path} 表格进度", unit="页"):
            page = pdf.pages[idx]
            page_num = idx + 1
            tables = page.extract_tables(filtered_settings) if filtered_settings else page.extract_tables()
            for table_idx, table in enumerate(tables):
                md_text = table_to_markdown(table, caption=f"第{page_num}页 表格{table_idx + 1}")
                if md_text:
                    table_chunks.append(md_text)
    return table_chunks


def extract_and_clean_all_pdf(pdf_config):
    all_chunks = []
    chunk_sources = []

    for config in pdf_config:
        pdf_name = config["pdf_path"].split("/")[-1]

        year = config.get("year", "未知")
        publisher = config.get("publisher", "未知")
        doc_type = config.get("doc_type", "未知")
        metadata_header = f"发布时间：{year}\n发布单位：{publisher}\n文件类型：{doc_type}\n正文："

        text = extract_text_from_pdf(config["pdf_path"], config["pages"])
        cleaned_text = apply_rules(text, config["rules"])
        tokenizer = load_auto_tokenizer(
            model_name=EMBEDDING_MODEL_NAME,
            token=HF_TOKEN,
        )
        chunks = split_by_tokens(
            cleaned_text,
            tokenizer,
            max_tokens=MAX_NEW_TOKENS,
            overlap=int(MAX_NEW_TOKENS * 0.2),
        )

        chunks = [metadata_header + chunk for chunk in chunks]

        all_chunks.extend(chunks)
        chunk_sources.extend([pdf_name] * len(chunks))

        if config.get("extract_tables", False):
            table_settings = config.get("table_settings", {})
            table_chunks = extract_tables_from_pdf(config["pdf_path"], config["pages"], table_settings)
            table_chunks = [metadata_header + chunk for chunk in table_chunks]
            all_chunks.extend(table_chunks)
            chunk_sources.extend([pdf_name] * len(table_chunks))

    return all_chunks, chunk_sources

if __name__ == "__main__":
    from config.settings import PDF_CONFIG

    idx=2
    pdf_i=PDF_CONFIG()[idx]

    text=extract_text_from_pdf(pdf_i["pdf_path"], pdf_i["pages"])
    cleaned_text=apply_rules(text, pdf_i["rules"])
    # print(text)
    # print(cleaned_text)
    tokenizer = load_auto_tokenizer(
        model_name=EMBEDDING_MODEL_NAME,
        token=HF_TOKEN,
    )
    chunks = split_by_tokens(
        cleaned_text,
        tokenizer,
        max_tokens=MAX_NEW_TOKENS,
        overlap=int(MAX_NEW_TOKENS * 0.2),
    )
    for i in range(len(chunks)):
        print(f"--- Chunk {i+1} ---")
        print(chunks[i])
        print("\n")
