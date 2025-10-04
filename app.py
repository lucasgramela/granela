"""
HabilitaIA - backend final para protótipo

- Dois passos:
  1) /edital/analisar -> recebe texto do edital (ou pdf b64), extrai se preciso, aplica heurística para listar documentos exigidos
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

IMPORTANTE: adapte endpoint DEEPSEEK_URL e payload conforme a API real que você contratar.
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
    import requests
except Exception:
    requests = None

# App
app = FastAPI(title="HabilitaIA - Final")

# Config
API_KEY = os.environ.get("API_KEY", "ramilo123")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.example/v1/analyze")
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

# normalize docs list
DOCS: List[Dict[str, Any]] = []
for d in _rules.get("doc_types", []):
    key = d.get("key") or d.get("name")
    DOCS.append({
        "key": key,
        "name": d.get("name", key),
        "category": d.get("category"),
        "keywords": [k.lower() for k in d.get("common_keywords", []) if isinstance(k, str)],
        "regex_examples": d.get("extraction", {}).get("regex_examples", {}),
        "obrigatorio": bool(d.get("obrigatorio", None) or (d.get("category") in ("juridica","fiscal_trabalhista","proposta"))),
        "raw": d,
    })

# Models
class EditalAnalysisRequest(BaseModel):
    texto: Optional[str] = None
    pdf_base64: Optional[str] = None
    data_sessao: Optional[str] = None  # expected date of edital opening YYYY-MM-DD or DD/MM/YYYY
    cnpj_empresa: Optional[str] = None

class EditalAnalysisResponse(BaseModel):
    status: str
    session_id: str
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

# Utils
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

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

# Heuristics for edital parsing (step 1)
def identify_documents_from_edital(text: str) -> List[Dict[str, Any]]:
    """Return list of documents with status 'aguardando_upload' and metadata"""
    t = normalize_text(text)
    out = []
    for d in DOCS:
        matched_kw = None
        for kw in d["keywords"]:
            if kw in t:
                matched_kw = kw
                break
        out.append({
            "key": d["key"],
            "name": d["name"],
            "category": d.get("category"),
            "status": "aguardando_upload",
            "matched_keyword": matched_kw,
            "obrigatorio": bool(doc.get("obrigatorio", False)),
        })
    return out

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
    extracted = {}
    reasons = []
    result = "amarelo"

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
        # generic rule: look for keywords in document
        doc_def = next((d for d in DOCS if d["key"] == doc_key), None)
        if doc_def:
            found = False
            for kw in doc_def.get("keywords", []):
                if kw in text_norm:
                    found = True
                    break
            if found:
                result = "verde"
            else:
                result = "amarelo"
                reasons.append("Conteúdo esperado não localizado via heurística")
        else:
            result = "amarelo"
            reasons.append("Documento não reconhecido nas regras")

    return {"result": result, "extracted": extracted, "reasons": reasons}

# DeepSeek integration (minimal, token-conscious)
def call_deepseek_api(api_key: str, prompt: str, max_tokens: int = 150) -> Optional[Dict[str, Any]]:
    if not requests:
        return {"error": "requests não disponível"}
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY não configurada"}
    payload = {
        "input": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": "json"  # ask for structured output if supported
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=15)
        if r.status_code != 200:
            return {"error": f"DeepSeek API retornou {r.status_code}", "text": r.text}
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Endpoints
@app.post("/edital/analisar", response_model=EditalAnalysisResponse)
def edital_analisar(req: EditalAnalysisRequest, request: Request):
    # api key check
    header_key = request.headers.get("x-api-key")
    if not header_key or header_key != API_KEY:
        raise HTTPException(status_code=401, detail="x-api-key inválida ou ausente")

    # get text
    text = ""
    if req.pdf_base64:
        try:
            pdf_bytes = base64.b64decode(req.pdf_base64)
            text = extract_text_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro extraindo PDF: {e}")
    else:
        text = req.texto or ""

    # identify documents
    docs = identify_documents_from_edital(text)

    # create session
    session_id = make_session_id()
    session = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_sessao": req.data_sessao,
        "cnpj_empresa": req.cnpj_empresa,
        "raw_edital_text_snippet": text[:4000],
        "documentos": docs,
        "cache": {},  # hash -> validation result
    }
    save_session(session)

    return EditalAnalysisResponse(status="ok", session_id=session_id, data_sessao=req.data_sessao, documentos=docs)

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

    if need_ai and DEEPSEEK_API_KEY and requests:
        # prepare small prompt/snippet
        snippet = find_relevant_snippets(text_input or (session.get("raw_edital_text_snippet","")), keywords=DOCS[0].get("keywords",[]))
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
