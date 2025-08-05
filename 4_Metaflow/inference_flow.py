# inference_flow.py
from metaflow import FlowSpec, step, Parameter
import os

class RadiologyExtractionFlow(FlowSpec):

    # ─── PARAMETERS ──────────────────────────────────────────────────────
    model_path  = Parameter(
        "model_path",
        default="../3_Fine_Tuning/fine_tuned_model/checkpoint-288",
        help="Path to your existing fine-tuned Bert token-classifier"
    )
    batch_size  = Parameter("batch_size",  default=25, help="How many reports per batch")
    delay_secs  = Parameter("delay_secs", default=60, help="Delay between batches (s)")

    @step
    def start(self):
        # Load env & raw data
        from dotenv import load_dotenv
        load_dotenv()
        assert os.getenv("GROQ_API_KEY"), "Missing GROQ_API_KEY in environment"

        import pandas as pd
        self.df = (
            pd.read_csv("data/raw/open_ave_data.csv")
              [["ReportText"]]
              .dropna()
              .reset_index(drop=True)
        )
        # If you already have extracted_results.csv, skip prompt_extraction:
        # self.next(self.prepare_dataset)
        self.next(self.prompt_extraction)

    @step
    def prompt_extraction(self):
        # ─── EXTRACT with GPT ────────────────────────────────────────────────
        from dotenv import load_dotenv
        load_dotenv()

        import nest_asyncio, asyncio
        from time import sleep
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        from langchain_core.output_parsers import PydanticOutputParser

        nest_asyncio.apply()

        class FieldsExtraction(BaseModel):
            findings:     str = Field("", description="Radiologist's observations")
            clinicaldata: str = Field("", description="Reason for exam")
            ExamName:     str = Field("", description="Exam type and date")
            impression:   str = Field("", description="Final diagnosis or summary")

        parser           = PydanticOutputParser(pydantic_object=FieldsExtraction)
        fmt_instructions = parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_messages([
            ("system",
"""You are a helpful medical data extraction assistant.

From the given "Report Text", extract the following fields and return ONLY a JSON object, and nothing else.
Use the format described in {format_instructions}.
Only return flat strings; if a field is missing return an empty string.
"""
            ),
            ("user", "{input}")
        ])

        llm   = ChatGroq(model="llama-3.1-8b-instant")
        chain = prompt | llm | parser
        texts = self.df["ReportText"].tolist()

        async def process(txt):
            try:
                return await chain.ainvoke({
                    "input": txt,
                    "format_instructions": fmt_instructions
                })
            except Exception as e:
                return None

        # throttle to avoid 429s
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_res = asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*(process(t) for t in batch))
            )
            results.extend(batch_res)
            print(f"Processed {i+len(batch)}/{len(texts)}; sleeping {self.delay_secs}s…")
            sleep(self.delay_secs)

        import pandas as pd
        parsed = [r.dict() if r else {"findings": "", "clinicaldata": "", "ExamName": "", "impression": ""} 
                  for r in results]
        pd.DataFrame(parsed).to_csv("extracted_results.csv", index=False)
        print("✅ extracted_results.csv saved")

        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self):
        # ─── BUILD TOKEN-CLASSIFICATION DS ───────────────────────────────────
        import pandas as pd
        from transformers import AutoTokenizer, AutoConfig
        from datasets import Dataset
        import re

        df_text = self.df
        df_lbl  = pd.read_csv("extracted_results.csv")

        # Load the exact tokenizer & labels you trained with
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        cfg = AutoConfig.from_pretrained(self.model_path)
        # id2label: {0:"O",1:"B-findings",...}
        self.label_list = [cfg.id2label[i] for i in range(len(cfg.id2label))]

        def char_tags(text, spans):
            tags = ["O"] * len(text)
            for fld, span in spans.items():
                if not span or not isinstance(span, str): continue
                for m in re.finditer(re.escape(span), text, flags=re.IGNORECASE):
                    for pos in range(m.start(), m.end()):
                        tags[pos] = fld
            return tags

        def tokenize_and_align(text, char_tags):
            enc = self.tokenizer(text, return_offsets_mapping=True, truncation=True)
            lbls, prev = [], "O"
            ct = char_tags
            for (s,e) in enc.offset_mapping:
                if s==e:
                    lbls.append("O")
                else:
                    c = ct[s]
                    if c=="O":
                        lbls.append("O")
                    else:
                        prefix = "B" if c!=prev else "I"
                        lbls.append(f"{prefix}-{c}")
                    prev = c
            enc["labels"] = [self.label_list.index(x) for x in lbls]
            return enc

        records = []
        for text, row in zip(df_text.ReportText, df_lbl.itertuples(index=False)):
            spans = {
                "ExamName":     row.ExamName,
                "clinicaldata": row.clinicaldata,
                "findings":     row.findings,
                "impression":   row.impression
            }
            tags = char_tags(text, spans)
            records.append(tokenize_and_align(text, tags))

        ds = Dataset.from_list(records).train_test_split(test_size=0.1, seed=42)
        self.train_ds, self.eval_ds = ds["train"], ds["test"]
        print(f"▶️  Built dataset: {len(self.train_ds)} train / {len(self.eval_ds)} eval")
        self.next(self.fine_tune)

    @step
    def fine_tune(self):
        # ─── FINE-TUNE (loads your existing head) ────────────────────────────
        import torch; assert torch.__version__ >= "2.6"
        from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
        import numpy as np
        from seqeval.metrics import classification_report

        # load your 13-label model!
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)

        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        args = TrainingArguments(
            output_dir="model_artifacts",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            evaluation_strategy="epoch",
            logging_steps=50,
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        def compute_metrics(p):
            preds  = np.argmax(p.predictions, axis=-1)
            labels = p.label_ids
            true_seq = [
                [ self.label_list[i] 
                  for (i,m) in zip(row, mask) if m ]
                for row, mask in zip(labels, p.prediction_mask)
            ]
            pred_seq = [
                [ self.label_list[i] 
                  for (i,m) in zip(row, mask) if m ]
                for row, mask in zip(preds, p.prediction_mask)
            ]
            print(classification_report(true_seq, pred_seq, digits=4))
            return {}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model("model_artifacts/final")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("✅ Fine-tuning complete. Artifacts under model_artifacts/")
        self.next(self.deploy)

    @step
    def deploy(self):
        print("📦 Ready for deployment (e.g., Docker, Streamlit)…")
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Pipeline finished!")

if __name__ == "__main__":
    RadiologyExtractionFlow()
