"""
HabilitaIA - backend final para protótipo

- Dois passos:
  1) /edital/analisar -> recebe texto do edital (ou pdf b64), extrai se preciso e usa IA para listar documentos exigidos
     -> salva sessão temporária em /tmp/habilitaia_sessions/<session>.json com os parâmetros extraídos (data_sessao, cnpj, lista esperada etc.)
     -> retorna session_id e lista de documentos com status "aguardando_upload"

  2) /documento/validar -> valida um documento por vez (json com session_id + doc_key + pdf_base64 OR texto)
     -> primeiro valida por heurística local (regex para datas, CNPJ, textos padronizados)
     -> se heurística insuficiente, chama DeepSeek (via API_KEY em DEEPSEEK_API_KEY) enviando apenas trecho relevante
     -> caching por hash do arquivo: se já validado, retorna resultado salvo
     -> responde com status: verde/amarelo/vermelho, extracted_fields, reasons

- Segurança: x-api-key header obrigatório (env API_KEY, fallback 'ramilo123')
- CORS habilitado por padrão
- Regras (habilitaia_rules.json) lidas se disponíveis


"""

import os
import io
import json
import base64
import hashlib
import zipfile
import tempfile
import re
import unicodedata
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel


# Optional libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# App
app = FastAPI(title="HabilitaIA - Final")

logger = logging.getLogger(__name__)


def _load_env_file(path: str = ".env") -> None:
    """Populate os.environ com variáveis declaradas em um arquivo .env simples."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


# Config
API_KEY = os.environ.get("API_KEY", "ramilo123")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-424f477f6fc546609b83975eb87342df")
DEEPSEEK_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")  # ou "deepseek-reasoner", se preferir
ENABLE_CORS = os.environ.get("ENABLE_CORS", "1")
SESSION_DIR = os.environ.get("SESSION_DIR", "/tmp/habilitaia_sessions")
FILES_DIR = os.environ.get("FILES_DIR", "/tmp/habilitaia_files")
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))  # not used now, placeholder

os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

# Load ruleset
RULES_PATHS = ["habilitaia_rules.json", "/mnt/data/habilitaia_rules.json"]
_rules = None
for p in RULES_PATHS:
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                _rules = json.load(f)
            break
        except Exception:
            _rules = None

if not _rules:
    # minimal fallback
    _rules = {
        "version": "fallback",
        "doc_types": [
            {"key": "cartao_cnpj", "name": "Comprovante de Inscrição no CNPJ", "category": "juridica", "common_keywords": ["cnpj"]},
            {"key": "contrato_social", "name": "Contrato Social", "category": "juridica", "common_keywords": ["contrato social", "estatuto"]},
            {"key": "cnd_federal", "name": "Certidão Negativa Federal", "category": "fiscal_trabalhista", "common_keywords": ["certidao negativa", "certidão negativa"]},
        ]
    }

# Utils
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def strip_accents(text: Optional[str]) -> str:
    """Return text without diacritics while preserving original casing."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _is_uppercase_heading(line: str) -> bool:
    """Heuristic to detect uppercase headings that delimit edital sections."""
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    normalized = strip_accents(stripped)
    if not any(ch.isalpha() for ch in normalized):
        return False
    return normalized == normalized.upper()



def extract_documentos_habilitacao_section(texto: str) -> str:
    """Return only the "DOCUMENTOS DE HABILITAÇÃO" block (up to next heading or 4000 chars)."""
    if not texto:
        return ""

    texto_sem_acentos = strip_accents(texto)
    texto_normalizado = texto_sem_acentos.upper()

    patterns = [
        re.compile(r"\bDOCUMENTOS?\s+(?:DE|PARA|DA|DO)\s+HABILITACAO\b"),
        re.compile(r"\bDOCUMENTACAO\s+(?:DE|PARA|DA|DO)\s+HABILITACAO\b"),
        re.compile(r"\bHABILITACAO\s+DOCUMENTOS?\b"),
    ]

    match = None
    for pattern in patterns:
        match = pattern.search(texto_normalizado)
        if match:
            break

    if not match:
        return ""

    start_idx = match.start()
    subseq = texto[start_idx:]
    max_len = 4000
    collected_lines: List[str] = []
    current_len = 0
    first_line = True

    for line in subseq.splitlines(True):
        if not first_line and _is_uppercase_heading(line):
            break
        collected_lines.append(line)
        current_len += len(line)
        if current_len >= max_len:
            break
        first_line = False

    section = "".join(collected_lines)
    if len(section) > max_len:
        section = section[:max_len]
    return section.strip()


# normalize docs list
DOCS: List[Dict[str, Any]] = []
for d in _rules.get("doc_types", []):
    key = d.get("key") or d.get("name")
    keywords = [k.lower() for k in d.get("common_keywords", []) if isinstance(k, str)]
    # also include normalized variations of the document name to improve recall
    name_tokens = [tok for tok in normalize_text(d.get("name", "")).split() if len(tok) >= 4]
    for token in name_tokens:
        if token and token not in keywords:
            keywords.append(token)
    DOCS.append({
        "key": key,
        "name": d.get("name", key),
        "category": d.get("category"),
        "keywords": keywords,
        "regex_examples": d.get("extraction", {}).get("regex_examples", {}),
        "obrigatorio": bool(d.get("obrigatorio", None) or (d.get("category") in ("juridica", "fiscal_trabalhista", "proposta"))),
        "raw": d,
    })

def normalize_ai_documents(ai_payload: Any) -> List[Dict[str, Any]]:
    """Normalize arbitrary IA payloads into a sanitized list of documents."""

    def _extract_payload(data: Any) -> Any:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return []
        if isinstance(data, dict):
            # Nested outputs such as {"output": {...}} or {"result_json": {...}}
            for key in ("result_json", "output", "data"):
                inner = data.get(key)
                if isinstance(inner, (dict, list)):
                    extracted = _extract_payload(inner)
                    if extracted:
                        return extracted
            for key in ("documents", "documentos", "docs", "itens", "items"):
                docs = data.get(key)
                if isinstance(docs, list):
                    return docs
            # If the dict itself resembles a document, wrap it
            if {"key", "name"}.issubset(set(data.keys())):
                return [data]
            return []
        if isinstance(data, list):
            return data
        return []

    raw_documents = _extract_payload(ai_payload)
    if not isinstance(raw_documents, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for item in raw_documents:
        if not isinstance(item, dict):
            continue
        # Work on a shallow copy so we do not mutate caller-provided structures
        doc = dict(item)

        # Resolve document key (accept alternate aliases and keep whatever the IA provided)
        key = doc.get("key") or doc.get("id") or doc.get("codigo") or doc.get("code")
        if key is None:
            # no identifiable key -> skip silently
            continue
        doc.setdefault("key", str(key))
        if not isinstance(doc["key"], str):
            doc["key"] = str(doc["key"])

        # Name/description: keep original naming but ensure "name" exists for compatibility
        if "name" not in doc:
            alt_name = doc.get("nome") or doc.get("descricao") or doc.get("description")
            if alt_name:
                doc["name"] = alt_name

        # Category: accept "categoria" while preserving original field names
        if "category" not in doc and doc.get("categoria") is not None:
            doc["category"] = doc["categoria"]

        # Status defaults to aguardando_upload if IA omits it
        doc.setdefault("status", "aguardando_upload")

        parsed.append(doc)

    return parsed


# Models
class EditalAnalysisRequest(BaseModel):
    texto: Optional[str] = None
    pdf_base64: Optional[str] = None
    data_sessao: Optional[str] = None  # expected date of edital opening YYYY-MM-DD or DD/MM/YYYY
    cnpj_empresa: Optional[str] = None

class EditalAnalysisResponse(BaseModel):
    status: str
    session_id: str
    numero_licitacao: Optional[str] = None
    municipio: Optional[str] = None
    data_sessao: Optional[str]
    documentos: List[Dict[str, Any]]

class DocumentValidateRequest(BaseModel):
    session_id: str
    doc_key: str
    pdf_base64: Optional[str] = None
    texto: Optional[str] = None

class DocumentValidateResponse(BaseModel):
    status: str
    key: str
    result: str  # verde|amarelo|vermelho
    extracted_fields: Dict[str, Any]
    reasons: List[str]
    cache_hit: bool = False

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber não instalado")
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

# Session helpers
def make_session_id() -> str:
    return hashlib.sha1(os.urandom(32)).hexdigest()

def session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def save_session(session: Dict[str, Any]):
    p = session_path(session["session_id"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

def load_session(session_id: str) -> Dict[str, Any]:
    p = session_path(session_id)
    if not os.path.exists(p):
        raise FileNotFoundError("Sessão não encontrada")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# IA para identificação de documentos no edital (step 1)
def identify_documents_with_ai(text: str) -> List[Dict[str, Any]]:
    """Consulta a IA para identificar documentos mencionados no edital."""
    if not text or not text.strip():
        return []

    if not OpenAI or not DEEPSEEK_API_KEY:
        return []

    prompt = {
        "task": "identify_required_documents",
        "instructions": (
            "Analise o texto do edital e retorne uma lista JSON com os documentos "
            "necessários. Cada item deve conter ao menos os campos 'key', 'name', "
            "'category' (opcional), 'status'='aguardando_upload' e 'obrigatorio' (bool)."
        ),
        "text": text[:8000],
    }

    ai_out = call_deepseek_api(DEEPSEEK_API_KEY, json.dumps(prompt), max_tokens=400)
    if not ai_out or ai_out.get("error"):
        return []

    def _coerce_list(value: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return None

    parsed: Optional[List[Dict[str, Any]]] = None
    if isinstance(ai_out, dict):
        for key in ("documents", "result", "output", "data"):
            parsed = _coerce_list(ai_out.get(key))
            if parsed is not None:
                break
        if parsed is None:
            text_field = ai_out.get("text") or ai_out.get("output_text")
            if isinstance(text_field, str):
                try:
                    parsed_candidate = json.loads(text_field)
                    parsed = _coerce_list(parsed_candidate)
                except Exception:
                    parsed = None

    return parsed or []


def identify_documents_with_ai(section_text: str) -> List[Dict[str, Any]]:
    """Use DeepSeek API to classify required documents from the extracted section."""
    if not section_text or not section_text.strip():
        return []

    if not OpenAI or not DEEPSEEK_API_KEY:
        logger.warning("IA indisponível para identificar documentos (openai_sdk=%s, api_key=%s)", bool(OpenAI), bool(DEEPSEEK_API_KEY))
        return []

    catalog = [
        {
            "key": d["key"],
            "name": d.get("name", d["key"]),
            "obrigatorio": bool(d.get("obrigatorio", False)),
        }
        for d in DOCS
    ]

    prompt = {
        "task": "identify_edital_documents",
        "catalog": catalog,
        "section": section_text.strip()[:4000],
        "instructions": "Leia a seção do edital e retorne JSON {'documentos': [{'key': str, 'status': 'aguardando_upload', 'obrigatorio': bool, 'justificativa': opcional}]}. Ignore documentos fora do catálogo.",
    }

    prompt_text = json.dumps(prompt, ensure_ascii=False)

    ai_out = call_deepseek_api(DEEPSEEK_API_KEY, prompt_text, max_tokens=200)
    if not ai_out or ai_out.get("error"):
        logger.warning("Falha ao consultar IA para documentos de habilitação: %s", ai_out.get("error") if isinstance(ai_out, dict) else "unknown")
        return []

    parsed: Optional[Dict[str, Any]] = None
    if isinstance(ai_out, dict):
        if isinstance(ai_out.get("result_json"), dict):
            parsed = ai_out.get("result_json")
        elif isinstance(ai_out.get("output"), dict):
            parsed = ai_out.get("output")
        else:
            text_candidate = None
            for key in ("text", "output_text", "output"):
                value = ai_out.get(key)
                if isinstance(value, str):
                    text_candidate = value
                    break
            if text_candidate:
                try:
                    parsed = json.loads(text_candidate)
                except Exception:
                    logger.warning("Não foi possível interpretar resposta textual da IA")
                    return []

    if not parsed or not isinstance(parsed, dict):
        logger.warning("Resposta da IA em formato inesperado: %s", ai_out)
        return []

    documentos_brutos = parsed.get("documentos")
    if not isinstance(documentos_brutos, list):
        logger.warning("IA não retornou lista de documentos válida")
        return []

    doc_map = {d["key"]: d for d in DOCS}
    seen: set = set()
    documentos_processados: List[Dict[str, Any]] = []

    for item in documentos_brutos:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not key or key not in doc_map or key in seen:
            continue
        seen.add(key)
        doc_def = doc_map[key]
        entry = {
            "key": key,
            "name": item.get("name") or doc_def.get("name") or key,
            "category": doc_def.get("category"),
            "status": item.get("status") or "aguardando_upload",
            "obrigatorio": bool(item.get("obrigatorio", doc_def.get("obrigatorio", False))),
        }
        justificativa = item.get("justificativa")
        if justificativa:
            entry["ia_justificativa"] = justificativa
        if item.get("confidence") is not None:
            entry["ia_confidence"] = item.get("confidence")
        documentos_processados.append(entry)

    return documentos_processados


# Patterns to capture licitação metadata
LICITACAO_PATTERNS = [
    re.compile(r"(?:processo(?:\s+licitat[óo]rio)?|preg[aã]o(?: eletr[ôo]nico)?|concorr[êe]ncia|tomada de preços|dispensa(?: de licita[çc][ãa]o)?|edital)\s*(?:n[ºo°\.\-]?\s*)?(\d{1,4}/\d{4})", re.IGNORECASE),
    re.compile(r"licita[çc][ãa]o\s*(?:n[ºo°\.\-]?\s*)?(\d{1,4}/\d{4})", re.IGNORECASE),
]

CITY_PATTERNS = [
    re.compile(r"munic[ií]pio de\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÖØ-öø-ÿ'\-\s]{2,})", re.IGNORECASE),
    re.compile(r"prefeitura municipal de\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÖØ-öø-ÿ'\-\s]{2,})", re.IGNORECASE),
]

VALOR_PATTERN = re.compile(
    r"(?:valor(?:\s+estimado)?|estimado em|montante)\s*(?:de|do|da|no)?\s*(R\$\s?[\d\.,]+)",
    re.IGNORECASE,
)

MODALIDADE_PATTERNS = [
    re.compile(r"modalidade\s*[:\-]?\s*([A-Za-zÀ-ÖØ-öø-ÿ'\s]{3,60})", re.IGNORECASE),
    re.compile(
        r"(preg[aã]o eletr[ôo]nico|preg[aã]o presencial|concorr[êe]ncia p[úu]blica|tomada de preços|dispensa de licita[çc][ãa]o)",
        re.IGNORECASE,
    ),
]

OBJETO_PATTERNS = [
    re.compile(r"objeto\s*[:\-]\s*(.{20,220})", re.IGNORECASE | re.DOTALL),
    re.compile(r"tem por objeto\s*(.{20,220})", re.IGNORECASE | re.DOTALL),
]


def extract_numero_licitacao(text: str) -> Optional[str]:
    for pattern in LICITACAO_PATTERNS:
        match = pattern.search(text)
        if match and match.group(1):
            return match.group(1).strip()
    return None


def extract_municipio(text: str) -> Optional[str]:
    for pattern in CITY_PATTERNS:
        match = pattern.search(text)
        if match and match.group(1):
            value = match.group(1).strip()
            # Normalize spacing but keep original casing for readability
            value = re.sub(r"\s+", " ", value)
            return value
    return None


def extract_valor_estimado(text: str) -> Optional[str]:
    match = VALOR_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_modalidade(text: str) -> Optional[str]:
    for pattern in MODALIDADE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            value = re.sub(r"\s+", " ", value)
            return value
    return None


def extract_objeto(text: str) -> Optional[str]:
    for pattern in OBJETO_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            value = value.split("\n")[0]
            if len(value) > 200:
                value = value[:200].rstrip() + "…"
            return value
    return None


SUMMARY_FIELD_ORDER = [
    "numero_licitacao",
    "data_sessao",
    "municipio",
    "modalidade",
    "objeto",
    "valor_estimado",
    "cnpj_empresa",
]
SUMMARY_FIELD_MAX_CHARS = 160
SUMMARY_TOTAL_MAX_CHARS = 600

DOC_SECTION_KEYWORDS = [
    r"documentos exigidos",
    r"documenta[çc][ãa]o exigida",
    r"habilita[çc][ãa]o",
    r"documenta[çc][ãa]o complementar",
]
DOC_SECTION_MAX_CHARS = 1800
DOC_SECTION_PADDING = 200


def _truncate_value(value: Optional[str], max_chars: int) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def build_edital_summary(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Compacta metadados do edital para envio junto ao prompt da IA."""

    context = context or {}
    numero = context.get("numero_licitacao") or extract_numero_licitacao(text)
    municipio = context.get("municipio") or extract_municipio(text)
    data_sessao = context.get("data_sessao")
    modalidade = context.get("modalidade") or extract_modalidade(text)
    objeto = context.get("objeto") or extract_objeto(text)
    valor = context.get("valor_estimado") or extract_valor_estimado(text)
    cnpj = context.get("cnpj_empresa") or extract_cnpj(text)

    raw = {
        "numero_licitacao": numero,
        "data_sessao": data_sessao,
        "municipio": municipio,
        "modalidade": modalidade,
        "objeto": objeto,
        "valor_estimado": valor,
        "cnpj_empresa": cnpj,
    }

    summary: Dict[str, str] = {}
    total_chars = 0
    for field in SUMMARY_FIELD_ORDER:
        value = _truncate_value(raw.get(field), SUMMARY_FIELD_MAX_CHARS)
        if not value:
            continue
        projected_total = total_chars + len(field) + len(value)
        if projected_total > SUMMARY_TOTAL_MAX_CHARS:
            continue
        summary[field] = value
        total_chars = projected_total
    return summary


def extract_document_section(text: str, max_chars: int = DOC_SECTION_MAX_CHARS) -> str:
    """Retorna apenas o trecho mais relevante sobre documentos para envio à IA."""

    for pattern in DOC_SECTION_KEYWORDS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - DOC_SECTION_PADDING)
            end = min(len(text), start + max_chars)
            return text[start:end]
    return text[:max_chars]



# Helper to extract small relevant snippets to send to IA
RELEVANT_TERMS = [
    "validade", "emissao", "vencimento", "prazo", "data de validade", "índice", "liquidez", "patrimônio", "capital", "cnd", "cnpj",
]

def find_relevant_snippets(text: str, keywords: List[str], window: int = 400) -> str:
    text_norm = text
    hits = []
    for kw in keywords + RELEVANT_TERMS:
        try:
            for m in re.finditer(re.escape(kw), text_norm, flags=re.IGNORECASE):
                start = max(0, m.start() - window//2)
                end = min(len(text_norm), m.end() + window//2)
                hits.append(text_norm[start:end])
        except Exception:
            continue
    if not hits:
        # fallback: return the first 2000 chars
        return text_norm[:2000]
    # join unique snippets
    joined = "\n---\n".join(dict.fromkeys(hits))
    return joined[:2000]

def identify_documents_with_ai(
    text: str,
    summary: Dict[str, str],
    existing_docs: Optional[List[Dict[str, Any]]] = None,
    *,
    api_key: Optional[str] = None,
    api_caller=None,
    max_tokens: int = 320,
) -> Dict[str, Any]:
    """Utiliza IA para complementar a identificação de documentos.

    O payload enviado à IA contém:
      - ``summary``: metadados compactados (numero, data, município, modalidade, objeto, valor, CNPJ).
      - ``document_section``: apenas o trecho relevante sobre documentação, limitado a ``DOC_SECTION_MAX_CHARS``.
      - ``known_documents``: lista atual de documentos identificados heurísticamente.
      - ``instructions``: orientação para retornar JSON estruturado ``{"documents": [...]}``.
    """

    existing_docs = existing_docs or []
    prompt_summary = summary or {}
    snippet = extract_document_section(text)

    payload = {
        "task": "identify_required_documents",
        "summary": prompt_summary,
        "document_section": snippet,
        "known_documents": existing_docs,
        "instructions": (
            "Use o campo `summary` com metadados compactos do edital (numero, data, municipio, modalidade, objeto, valor, CNPJ). "
            "Analise exclusivamente o trecho `document_section` e responda JSON no formato {\"documents\": [{\"key\": str, \"name\": str, "
            "\"category\": str|null, \"obrigatorio\": bool, \"observacoes\": str?}]}."
        ),
    }

    api_key = DEEPSEEK_API_KEY if api_key is None else api_key
    caller = api_caller or call_deepseek_api


    def _error_payload(code: str, message: str, http_status: int = 503) -> Dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "http_status": http_status,
        }

    
    if not api_key:
        return {
            "used_ai": False,
            "documents": [],
            "raw_response": None,
            "prompt": payload,
            "error": _error_payload(
                "missing_api_key",
                "DeepSeek API não configurada",
            ),
        }
    if caller is call_deepseek_api and OpenAI is None:
        return {
            "used_ai": False,
            "documents": [],
            "raw_response": None,
            "prompt": payload,
            "error": _error_payload(
                "missing_openai_dependency",
                "Dependência 'openai' não disponível para chamadas à IA",
            ),
        }
    response = caller(api_key, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens)
    documents: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    
    if isinstance(response, dict):
        if response.get("error"):
            error = _error_payload("ai_error", str(response.get("error")), 502)
        candidate: Optional[Any] = None
        if isinstance(response.get("result_json"), dict):
            candidate = response.get("result_json")
        elif isinstance(response.get("output"), dict):
            candidate = response.get("output")
        elif isinstance(response.get("output"), list):
            candidate = {"documents": response.get("output")}
        else:
            for key in ("text", "output_text", "output"):
                txt = response.get(key)
                if isinstance(txt, str):
                    try:
                        candidate = json.loads(txt)
                        break
                    except Exception:
                        continue
        if isinstance(candidate, dict):
            docs_list = candidate.get("documents") or candidate.get("documentos")
            if isinstance(docs_list, list):
                for item in docs_list:
                    if isinstance(item, dict):
                        documents.append(item)
        elif error is None:
            error = _error_payload("unexpected_ai_response", "Resposta da IA em formato inesperado", 502)
    else:
        error = _error_payload("invalid_ai_response", "Resposta da IA inválida ou vazia", 502)

    if not documents and error is None:
        error = _error_payload("empty_ai_documents", "IA não retornou documentos identificados", 424)

    result = {
        "used_ai": True,
        "documents": documents,
        "raw_response": response,
        "prompt": payload,
    }
    if error:
        result["error"] = error
    return result


# Simple extraction examples (date, cnpj)
DATE_PATTERNS = [r"(\d{2}/\d{2}/\d{4})", r"(\d{4}-\d{2}-\d{2})"]
CNPJ_PATTERN = r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"

def extract_dates(text: str) -> List[str]:
    out = []
    for p in DATE_PATTERNS:
        for m in re.findall(p, text):
            out.append(m)
    return out

def extract_cnpj(text: str) -> Optional[str]:
    m = re.search(CNPJ_PATTERN, text)
    return m.group(0) if m else None

# Heuristic validator for a known document
def heuristic_validate(doc_key: str, text: str, session: Dict[str, Any]) -> Dict[str, Any]:
    text_norm = normalize_text(text)
    extracted: Dict[str, Any] = {}
    reasons: List[str] = []
    result = "amarelo"

    doc_def = next((d for d in DOCS if d["key"] == doc_key), None)
    if not doc_def:
        reasons.append("Documento não reconhecido nas regras")
        return {"result": result, "extracted": extracted, "reasons": reasons}

    doc_name = doc_def.get("name") or doc_key
    keywords = [kw for kw in doc_def.get("keywords", []) if kw]
    matched_keyword = None
    for kw in keywords:
        if kw in text_norm:
            matched_keyword = kw
            break

    if matched_keyword:
        extracted["matched_keyword"] = matched_keyword

    has_keywords = bool(keywords)
    recognized_by_keywords = bool(matched_keyword or not has_keywords)

    strict_mismatch_keys = {"cartao_cnpj", "cnd_federal", "crf_fgts", "cndt_trabalhista"}
    if doc_key.startswith("contrato"):
        strict_mismatch_keys.add(doc_key)

    if has_keywords and not recognized_by_keywords:
        reasons.append(
            f"Palavras-chave esperadas para '{doc_name}' não foram encontradas; o arquivo parece não corresponder ao tipo solicitado."
        )
        mismatch_result = "vermelho" if (doc_key in strict_mismatch_keys) else result
        return {"result": mismatch_result, "extracted": extracted, "reasons": reasons}

    # example: certidão (cnd_federal) -> look for "validade" or a date
    if doc_key in ("cnd_federal", "crf_fgts", "cndt_trabalhista"):
        dates = extract_dates(text)
        if dates:
            extracted["dates"] = dates
            # pick latest date and compare with session date
            try:
                # prefer format DD/MM/YYYY
                dlist = []
                for ds in dates:
                    if '/' in ds:
                        dlist.append(datetime.strptime(ds, "%d/%m/%Y"))
                    elif '-' in ds:
                        dlist.append(datetime.strptime(ds, "%Y-%m-%d"))
                if dlist:
                    latest = max(dlist)
                    extracted["representative_date"] = latest.isoformat()
                    if session.get("data_sessao"):
                        try:
                            # try parse session date (try YYYY-MM-DD then DD/MM/YYYY)
                            sd = session["data_sessao"]
                            if '-' in sd:
                                sess_date = datetime.strptime(sd, "%Y-%m-%d")
                            else:
                                sess_date = datetime.strptime(sd, "%d/%m/%Y")
                            if latest >= sess_date:
                                result = "verde"
                            else:
                                result = "vermelho"
                                reasons.append("Documento vencido em comparação com a data do edital")
                        except Exception:
                            # cannot parse session date
                            result = "amarelo"
                            reasons.append("Não foi possível comparar validade (formato de data inválido)")
                    else:
                        result = "amarelo"
                        reasons.append("Data do edital não informada para comparação")
                else:
                    result = "amarelo"
                    reasons.append("Não foi possível extrair data da certidão")
            except Exception as e:
                result = "amarelo"
                reasons.append(f"Erro ao processar datas: {e}")
        else:
            result = "amarelo"
            reasons.append("Nenhuma data encontrada na certidão")
    elif doc_key == "cartao_cnpj" or doc_key.startswith("contrato"):
        cnpj = extract_cnpj(text)
        if cnpj:
            extracted["cnpj_found"] = cnpj
            # compare with session cnpj if available
            if session.get("cnpj_empresa"):
                if normalize_text(session.get("cnpj_empresa")) in normalize_text(cnpj):
                    result = "verde"
                else:
                    result = "vermelho"
                    reasons.append("CNPJ do documento não confere com o CNPJ informado no edital")
            else:
                result = "verde"
        else:
            result = "amarelo"
            reasons.append("CNPJ não encontrado no documento")
    else:
        # generic rule: as palavras-chave já foram verificadas
        if recognized_by_keywords:
            result = "verde"
        else:
            result = "amarelo"
            reasons.append("Conteúdo esperado não localizado via heurística")

    return {"result": result, "extracted": extracted, "reasons": reasons}

# DeepSeek integration (minimal, token-conscious)
def call_deepseek_api(api_key: str, prompt: str, max_tokens: int = 150) -> Optional[Dict[str, Any]]:
    if OpenAI is None:
        return {"error": "SDK OpenAI não disponível"}
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY não configurada"}

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_URL or "https://api.deepseek.com",
        )
    except Exception as e:
        logger.exception("Falha ao inicializar cliente DeepSeek")
        return {"error": f"Falha ao inicializar cliente: {e}"}

    system_prompt = (
        "Você é um assistente especializado em licitações brasileiras. "
        "Responda estritamente em JSON válido."
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0,
            stream=False,
        )
    except Exception as e:
        logger.exception("Erro ao chamar DeepSeek")
        return {"error": str(e)}

    try:
        content = response.choices[0].message.content  # type: ignore[index]
    except (AttributeError, IndexError, KeyError):
        content = None

    if not content:
        return {"error": "Resposta vazia da DeepSeek"}

    try:
        parsed_content = json.loads(content)
    except Exception:
        return {"text": content}

    if isinstance(parsed_content, dict):
        return parsed_content

    return {"output": parsed_content}


# Endpoints
@app.post("/edital/analisar", response_model=EditalAnalysisResponse)
def edital_analisar(req: EditalAnalysisRequest, request: Request):
    # api key check
    header_key = request.headers.get("x-api-key")
    if not header_key or header_key != API_KEY:
        raise HTTPException(status_code=401, detail="x-api-key inválida ou ausente")

    # get text
    text = ""
    numero_licitacao: Optional[str] = None
    if req.pdf_base64:
        try:
            pdf_bytes = base64.b64decode(req.pdf_base64)
            text = extract_text_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro extraindo PDF: {e}")
    else:
        text = req.texto or ""

    if text:
        numero_licitacao = extract_numero_licitacao(text)
    municipio = extract_municipio(text)

    documentos_section = extract_documentos_habilitacao_section(text)
    if not documentos_section:
        documentos_section = extract_document_section(text)

    docs: List[Dict[str, Any]] = []
    normalized_text = normalize_text(documentos_section or text)
    for doc_def in DOCS:
        entry = {
            "key": doc_def.get("key"),
            "name": doc_def.get("name"),
            "category": doc_def.get("category"),
            "status": "aguardando_upload",
            "obrigatorio": doc_def.get("obrigatorio"),
            "matched_keyword": None,
        }
        keywords = [kw for kw in doc_def.get("keywords", []) if kw]
        matched_keyword = next((kw for kw in keywords if kw in normalized_text), None)
        if matched_keyword:
            entry["matched_keyword"] = matched_keyword
        docs.append(entry)

    # Build compact summary for prompts / logging
    summary_context = {
        "numero_licitacao": numero_licitacao,
        "municipio": municipio,
        "data_sessao": req.data_sessao,
        "cnpj_empresa": req.cnpj_empresa,
    }
    edital_summary = build_edital_summary(text, summary_context)
    trimmed_docs = [
        {
            "key": d.get("key"),
            "name": d.get("name"),
            "category": d.get("category"),
            "obrigatorio": d.get("obrigatorio"),
        }
        for d in docs
    ]
    ai_doc_result = identify_documents_with_ai(
        documentos_section or text,
        edital_summary,
        trimmed_docs,
        api_key=DEEPSEEK_API_KEY,
    )

    ai_docs = []
    used_ai = False
    ai_error = None
    if isinstance(ai_doc_result, dict):
        ai_docs = ai_doc_result.get("documents") or []
        used_ai = bool(ai_doc_result.get("used_ai"))
        ai_error = ai_doc_result.get("error")

    if not used_ai or not ai_docs:
        error_detail = {
            "message": "Não foi possível identificar documentos automaticamente.",
        }
        status_code = 503
        if isinstance(ai_error, dict):
            status_code = int(ai_error.get("http_status", status_code))
            if ai_error.get("message"):
                error_detail["reason"] = ai_error.get("message")
            if ai_error.get("code"):
                error_detail["code"] = ai_error.get("code")
        elif ai_error:
            error_detail["reason"] = str(ai_error)
        elif used_ai:
            status_code = 502
        logger.warning("Identificação automática indisponível: %s", error_detail)
        raise HTTPException(status_code=status_code, detail=error_detail)

    existing_keys = {d.get("key") for d in docs if d.get("key")}
    for suggestion in ai_docs:
        if not isinstance(suggestion, dict):
            continue
        key = suggestion.get("key") or suggestion.get("id")
        if not key or key in existing_keys:
            continue
        docs.append(

        {
            "key": key,
            "name": suggestion.get("name")
            or suggestion.get("nome")
            or key,
            "category": suggestion.get("category")
            or suggestion.get("categoria"),
            "status": "aguardando_upload",
            "matched_keyword": None,
            "obrigatorio": bool(suggestion.get("obrigatorio"))
            if suggestion.get("obrigatorio") is not None
            else False,
            "ai_suggestion": True,
            }
     
        )

    ai_docs_raw: Any = []
    if isinstance(ai_doc_result, dict):
        ai_docs_raw = ai_doc_result.get("documents") or []
    docs = normalize_ai_documents(ai_docs_raw)

    # create session
    session_id = make_session_id()
    session = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_sessao": req.data_sessao,
        "cnpj_empresa": req.cnpj_empresa,
        "numero_licitacao": numero_licitacao,
        "municipio": municipio,
        "modalidade": edital_summary.get("modalidade"),
        "valor_estimado": edital_summary.get("valor_estimado"),
        "objeto": edital_summary.get("objeto"),
        "edital_summary": edital_summary,
        "ai_document_identification": {
            "used_ai": ai_doc_result.get("used_ai", False),
        },
        "raw_edital_text_snippet": text[:4000],
        "documentos_section": documentos_section,
        "documentos": docs,
        "cache": {},  # hash -> validation result
    }
    if ai_doc_result.get("prompt"):
        session["ai_document_identification"]["prompt"] = ai_doc_result.get("prompt")
    if ai_doc_result.get("raw_response"):
        session["ai_document_identification"]["raw_response"] = ai_doc_result.get("raw_response")
    save_session(session)

    return EditalAnalysisResponse(
        status="ok",
        session_id=session_id,
        data_sessao=req.data_sessao,
        numero_licitacao=numero_licitacao,
        municipio=municipio,
        documentos=docs,
    )


@app.post("/documento/validar", response_model=DocumentValidateResponse)
def documento_validar(req: DocumentValidateRequest, request: Request):
    # api key check
    header_key = request.headers.get("x-api-key")
    if not header_key or header_key != API_KEY:
        raise HTTPException(status_code=401, detail="x-api-key inválida ou ausente")

    # load session
    try:
        session = load_session(req.session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    # accept either texto or pdf_base64
    content_bytes = None
    text_input = req.texto
    if req.pdf_base64:
        try:
            content_bytes = base64.b64decode(req.pdf_base64)
            # compute hash
            file_hash = sha256_bytes(content_bytes)
            # save file optionally
            fname = os.path.join(FILES_DIR, f"{file_hash}.pdf")
            if not os.path.exists(fname):
                with open(fname, "wb") as f:
                    f.write(content_bytes)
            # extract text (try pdfplumber)
            try:
                text_input = extract_text_from_pdf_bytes(content_bytes)
            except Exception:
                text_input = None
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro decodificando PDF: {e}")
    else:
        # no file
        content_bytes = None
        file_hash = None
        text_input = text_input or ""

    # check cache
    if file_hash and file_hash in session.get("cache", {}):
        cached = session["cache"][file_hash]
        # return cached response
        resp = DocumentValidateResponse(status="ok", key=req.doc_key, result=cached.get("result","amarelo"), extracted_fields=cached.get("extracted",{}), reasons=cached.get("reasons",[]), cache_hit=True)
        return resp

    # run heuristic first
    heur = heuristic_validate(req.doc_key, text_input or "", session)
    result = heur["result"]
    extracted = heur["extracted"]
    reasons = heur["reasons"]

    # decide if we need IA: only if heuristica inconclusiva (amarelo) or missing important fields
    need_ai = (result == "amarelo")

    if need_ai and DEEPSEEK_API_KEY and OpenAI:
        # prepare small prompt/snippet
        doc_def = next((d for d in DOCS if d["key"] == req.doc_key), None)
        doc_keywords = doc_def.get("keywords", []) if doc_def else []
        context_candidates = [
            text_input,
            session.get("documentos_section"),
            session.get("raw_edital_text_snippet", ""),
        ]
        context_source = next((c for c in context_candidates if c), "")
        snippet = find_relevant_snippets(context_source, keywords=doc_keywords)
        prompt = {
            "task": "validate_document",
            "doc_key": req.doc_key,
            "snippet": snippet,
            "session_data": {"data_sessao": session.get("data_sessao"), "cnpj_empresa": session.get("cnpj_empresa")},
            "instructions": "Retorne um JSON com fields: result=['verde'|'amarelo'|'vermelho'], extracted_fields (dict), reasons (array of strings). Seja sucinto. max_tokens=150"
        }
        ai_out = call_deepseek_api(DEEPSEEK_API_KEY, json.dumps(prompt), max_tokens=150)
        if ai_out and isinstance(ai_out, dict) and not ai_out.get("error"):
            # try to parse structured response (this depends on DeepSeek implementation)
            # we expect ai_out to contain `output` or `result_json`
            parsed = None
            if ai_out.get("result_json"):
                parsed = ai_out.get("result_json")
            elif ai_out.get("output") and isinstance(ai_out.get("output"), dict):
                parsed = ai_out.get("output")
            else:
                # try to parse text
                txt = ai_out.get("text") or ai_out.get("output_text") or ai_out.get("output")
                if isinstance(txt, str):
                    try:
                        parsed = json.loads(txt)
                    except Exception:
                        parsed = None
            if parsed:
                # merge
                result = parsed.get("result", result)
                extracted.update(parsed.get("extracted_fields", {}))
                reasons += parsed.get("reasons", [])
        else:
            reasons.append("IA inacessível ou falha: " + str(ai_out.get("error") if ai_out else "unknown"))

    # finalize
    # save to cache if hash available
    cache_entry = {"result": result, "extracted": extracted, "reasons": reasons, "at": datetime.utcnow().isoformat()}
    if file_hash:
        session.setdefault("cache", {})[file_hash] = cache_entry
        save_session(session)

    return DocumentValidateResponse(status="ok", key=req.doc_key, result=result, extracted_fields=extracted, reasons=reasons, cache_hit=False)

@app.get("/sessao/{session_id}")
def sessao_get(session_id: str, request: Request):
    header_key = request.headers.get("x-api-key")
    if not header_key or header_key != API_KEY:
        raise HTTPException(status_code=401, detail="x-api-key inválida ou ausente")
    try:
        s = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return s

@app.post("/sessao/{session_id}/zip")
def sessao_zip(session_id: str, request: Request):
    header_key = request.headers.get("x-api-key")
    if not header_key or header_key != API_KEY:
        raise HTTPException(status_code=401, detail="x-api-key inválida ou ausente")
    try:
        s = load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    # gather validated green docs from cache
    files_to_zip = []
    manifest = {"session_id": session_id, "generated_at": datetime.utcnow().isoformat(), "documents": []}
    cache = s.get("cache", {})
    for h, entry in cache.items():
        if entry.get("result") == "verde":
            # find file by hash
            fpath = os.path.join(FILES_DIR, f"{h}.pdf")
            if os.path.exists(fpath):
                files_to_zip.append((os.path.basename(fpath), fpath))
                manifest["documents"].append({"file": os.path.basename(fpath), "hash": h, "extracted": entry.get("extracted", {})})

    # create zip
    if not files_to_zip:
        raise HTTPException(status_code=400, detail="Nenhum documento validado (verde) para gerar zip")
    fd, zipname = tempfile.mkstemp(prefix=f"habilitaia_{session_id}_", suffix=".zip", dir=tempfile.gettempdir())
    os.close(fd)
    with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # add files
        for fname, path in files_to_zip:
            z.write(path, arcname=fname)
        # add manifest
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return FileResponse(zipname, media_type="application/zip", filename=os.path.basename(zipname))

@app.get("/")
def home():
    return {"message": "HabilitaIA API rodando. Endpoints: POST /edital/analisar , POST /documento/validar , GET /sessao/{id} , POST /sessao/{id}/zip"}
