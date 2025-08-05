# streamlit_app.py
import streamlit as st

# ─── Prompt-based extraction (Groq + LangChain) ─────────────────────────
from dotenv import load_dotenv
load_dotenv()
import nest_asyncio, asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
nest_asyncio.apply()

class FieldsExtraction(BaseModel):
    findings: str = Field("", description="Radiologist's observations")
    clinicaldata: str = Field("", description="Reason for exam")
    ExamName: str = Field("", description="Exam type and date")
    impression: str = Field("", description="Final diagnosis or summary")

_prompt_parser = PydanticOutputParser(pydantic_object=FieldsExtraction)
_prompt_fmt    = _prompt_parser.get_format_instructions()
_prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a helpful medical-data extraction assistant.
Extract *only* a flat JSON object with these fields: findings, clinicaldata, ExamName, impression.
Use exactly these instructions (don't invent extra keys):
{format_instructions}
If a field is missing in the report, return it as an empty string.""" ),
    ("user", "{input}")
])
_prompt_llm   = ChatGroq(model="llama-3.1-8b-instant")
_prompt_chain = _prompt_template | _prompt_llm | _prompt_parser

async def _prompt_extract_async(txt: str):
    return await _prompt_chain.ainvoke({
        "input": txt,
        "format_instructions": _prompt_fmt
    })

def run_prompt_extraction(txt: str):
    """Returns a dict with keys findings, clinicaldata, ExamName, impression."""
    try:
        out = asyncio.get_event_loop().run_until_complete(_prompt_extract_async(txt))
        return out.dict()
    except Exception:
        return {k:"" for k in FieldsExtraction.__fields__}

# ─── Fine-tuned extraction (HuggingFace pipeline) ───────────────────────
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
@st.cache_resource
def load_fine_tuned(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    # aggregation_strategy="simple" ties together multi-token entities
    ner = pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")
    # get label names in order (e.g. "B-findings", "I-findings", ...)
    id2label = mdl.config.id2label
    # strip leading B-/I- for grouping
    groups = list({lab.split("-",1)[1] for lab in id2label.values() if lab!="O"})
    return ner, groups

def run_fine_tuned_extraction(txt: str, model_dir: str):
    ner, groups = load_fine_tuned(model_dir)
    ents = ner(txt)
    res = {g: "" for g in groups}
    for ent in ents:
        g = ent["entity_group"]
        res[g] += ent["word"] + " "
    return {g: v.strip() for g,v in res.items()}

# ────────────────────────────────────────────────────────────────────────
st.title("🏥 Radiology Field Extraction")

mode = st.sidebar.radio("Choose model", ["Prompt-based Model", "Fine-tuned Model"])

report = st.text_area("📄 Paste your radiology report here", height=250)

if st.button("Extract Fields"):
    if not report.strip():
        st.warning("Please paste some text above first.")
        st.stop()
    if mode=="Prompt-based Model":
        with st.spinner("Running prompt extraction…"):
            out = run_prompt_extraction(report)
    else:
        model_dir = "/Users/aakash/Desktop/BuildProjects_OpenAvenue/3_Fine_Tuning/bert-radiology-token-classifier"
        with st.spinner("Running fine-tuned extraction…"):
            out = run_fine_tuned_extraction(report, model_dir)
    st.subheader("🔍 Extracted Fields")
    st.table(out)