"""
Closed-Loop Optimization Engine for AccaHumanize-CL.

Combines AI Humanizer (Generator), RoBERTa Detector (Critic), and Semantics Guardrail
into a closed-loop feedback pipeline for academic text refinement.
"""

import sys
import os
from typing import Dict, Any, List, Generator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.semantics_guardrail import SemanticsGuardrail
from src.highlighter import segment_sentences, run_predictions
from robustness_eval.humanizer.humanizer_engine import AIHumanizerEngine
from robustness_eval.rewriters import combined_rewrite

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Global singleton instance for humanizer engine
_HUMANIZER_ENGINE = None

def get_humanizer_engine() -> AIHumanizerEngine:
    global _HUMANIZER_ENGINE
    if _HUMANIZER_ENGINE is None:
        _HUMANIZER_ENGINE = AIHumanizerEngine(model_name_or_path="models/t5-humanizer-v1")
    return _HUMANIZER_ENGINE


class ClosedLoopHumanizer:
    def __init__(
        self, 
        model_dir: str = "models/modernbert-academic", 
        roberta_dir: str = "models/roberta-sentence-academic-v2",
        preloaded_model=None, 
        preloaded_tokenizer=None,
        preloaded_roberta_model=None,
        preloaded_roberta_tokenizer=None
    ):
        self.guardrail = SemanticsGuardrail()
        self.humanizer = get_humanizer_engine()
        self.model_dir = model_dir
        self.roberta_dir = roberta_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.preloaded_model = preloaded_model
        self.preloaded_tokenizer = preloaded_tokenizer
        self.preloaded_roberta_model = preloaded_roberta_model
        self.preloaded_roberta_tokenizer = preloaded_roberta_tokenizer

        # Load ModernBERT if not preloaded
        if self.preloaded_model is None and os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
            try:
                self.preloaded_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.preloaded_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
                self.preloaded_model.eval()
            except Exception as e:
                print(f"[closed_loop] ModernBERT loading warning: {e}")

        # Load RoBERTa if not preloaded
        if self.preloaded_roberta_model is None and os.path.exists(roberta_dir) and os.path.exists(os.path.join(roberta_dir, "config.json")):
            try:
                self.preloaded_roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_dir)
                self.preloaded_roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_dir).to(self.device)
                self.preloaded_roberta_model.eval()
            except Exception as e:
                print(f"[closed_loop] RoBERTa loading warning: {e}")

    def _predict_sentence_ai_prob(self, sentence: str) -> float:
        """Helper to compute AI probability for a single sentence via Hybrid Ensemble Engine."""
        try:
            preds, _ = run_predictions(
                [sentence],
                model_dir=self.model_dir,
                preloaded_model=self.preloaded_model,
                preloaded_tokenizer=self.preloaded_tokenizer,
                preloaded_roberta_model=self.preloaded_roberta_model,
                preloaded_roberta_tokenizer=self.preloaded_roberta_tokenizer,
                roberta_dir=self.roberta_dir
            )
            if preds and isinstance(preds, list) and len(preds) > 0:
                score = preds[0].get("score", preds[0].get("prob", 0.5))
                return float(score)
        except Exception as e:
            print(f"[closed_loop] Prediction error: {e}")
        return 0.5

    def _predict_batch_ai_probs(self, sentences: List[str]) -> List[float]:
        """Helper to compute AI probabilities for candidate sentences in a single batched tensor pass."""
        if not sentences:
            return []
        try:
            preds, _ = run_predictions(
                sentences,
                model_dir=self.model_dir,
                preloaded_model=self.preloaded_model,
                preloaded_tokenizer=self.preloaded_tokenizer,
                preloaded_roberta_model=self.preloaded_roberta_model,
                preloaded_roberta_tokenizer=self.preloaded_roberta_tokenizer,
                roberta_dir=self.roberta_dir
            )
            if preds and isinstance(preds, list):
                return [float(p.get("score", p.get("prob", 0.5))) for p in preds]
        except Exception as e:
            print(f"[closed_loop] Batch prediction error: {e}")
        return [0.5] * len(sentences)

    def process_closed_loop_stream(
        self,
        text: str,
        tone: str = "academic",
        intensity: float = 0.8,
        min_semantic_sim: float = 0.45
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Closed-loop streaming processor. Yields real-time step metadata for each sentence.
        """
        # Step 1: Token Locking
        locked_text, lock_dict = self.guardrail.lock_tokens(text)

        # Step 2: Sentence Segmentation
        sentences = segment_sentences(locked_text)
        total_sentences = len(sentences)

        yield {
            "type": "init",
            "total_sentences": total_sentences,
            "locked_tokens_count": len(lock_dict),
            "locked_tokens": list(lock_dict.values())
        }

        optimized_sentences = []
        overall_initial_ai_scores = []
        overall_final_ai_scores = []
        overall_sim_scores = []

        for idx, orig_sentence_locked in enumerate(sentences):
            orig_sentence = self.guardrail.restore_tokens(orig_sentence_locked, lock_dict)
            initial_ai_prob = self._predict_sentence_ai_prob(orig_sentence)
            overall_initial_ai_scores.append(initial_ai_prob)

            # Adaptive Early Exit: if sentence is already classified as natural human (< 15% AI score)
            if initial_ai_prob < 0.15:
                optimized_sentences.append(orig_sentence)
                overall_final_ai_scores.append(initial_ai_prob)
                overall_sim_scores.append(1.0)
                yield {
                    "type": "step",
                    "index": idx,
                    "total": total_sentences,
                    "original": orig_sentence,
                    "humanized": orig_sentence,
                    "initial_ai_prob": round(initial_ai_prob, 4),
                    "final_ai_prob": round(initial_ai_prob, 4),
                    "semantic_similarity": 1.0,
                    "citations_preserved": True
                }
                continue

            # Generate candidate pool
            neural_candidates = self.humanizer.generate_neural_candidates(
                sentence=orig_sentence_locked,
                tone=tone
            )
            rule_candidate = combined_rewrite(orig_sentence_locked, tone=tone)
            hybrid_candidates = [combined_rewrite(nc, tone=tone) for nc in neural_candidates]
            all_candidates = [rule_candidate] + neural_candidates + hybrid_candidates + [orig_sentence_locked]

            valid_candidates = []
            for cand_locked in all_candidates:
                cand_restored = self.guardrail.restore_tokens(cand_locked, lock_dict)
                is_valid, _ = self.guardrail.verify_academic_integrity(orig_sentence, cand_restored, lock_dict)
                if not is_valid:
                    continue
                sim = self.guardrail.calculate_semantic_similarity(orig_sentence, cand_restored)
                if sim < min_semantic_sim:
                    continue
                valid_candidates.append((cand_restored, sim))

            best_sentence = orig_sentence
            best_ai_prob = initial_ai_prob
            best_sim = 1.0

            if valid_candidates:
                cand_texts = [c[0] for c in valid_candidates]
                cand_probs = self._predict_batch_ai_probs(cand_texts)

                for (cand_restored, sim), cand_ai in zip(valid_candidates, cand_probs):
                    if cand_ai < best_ai_prob:
                        best_ai_prob = cand_ai
                        best_sentence = cand_restored
                        best_sim = sim

            optimized_sentences.append(best_sentence)
            overall_final_ai_scores.append(best_ai_prob)
            overall_sim_scores.append(best_sim)

            step_event = {
                "type": "step",
                "index": idx,
                "total": total_sentences,
                "original": orig_sentence,
                "humanized": best_sentence,
                "initial_ai_prob": round(initial_ai_prob, 4),
                "final_ai_prob": round(best_ai_prob, 4),
                "semantic_similarity": round(best_sim, 4),
                "citations_preserved": True
            }
            yield step_event

        # Final Summary
        final_doc = " ".join(optimized_sentences)
        avg_initial_ai = float(sum(overall_initial_ai_scores) / max(1, len(overall_initial_ai_scores)))
        avg_final_ai = float(sum(overall_final_ai_scores) / max(1, len(overall_final_ai_scores)))
        avg_sim = float(sum(overall_sim_scores) / max(1, len(overall_sim_scores)))

        yield {
            "type": "complete",
            "final_text": final_doc,
            "avg_initial_ai_score": round(avg_initial_ai, 4),
            "avg_final_ai_score": round(avg_final_ai, 4),
            "avg_semantic_similarity": round(avg_sim, 4),
            "total_sentences_processed": total_sentences
        }

    def process_closed_loop_tex_stream(
        self,
        tex_content: str,
        tone: str = "academic",
        min_semantic_sim: float = 0.45
    ) -> Generator[Dict[str, Any], None, None]:
        """
        LaTeX document-level streaming processor. Deconstructs .tex AST, locks math/tables/citations,
        optimizes body text paragraphs via closed-loop feedback, and yields SSE updates.
        """
        from src.tex_parser import LaTeXASTParser
        parser = LaTeXASTParser()

        locked_tex, lock_dict, paragraphs = parser.lock_tex(tex_content)
        total_paras = len(paragraphs)

        yield {
            "type": "tex_init",
            "total_paragraphs": total_paras,
            "locked_tokens_count": len(lock_dict),
            "locked_tokens": list(lock_dict.values())[:10]
        }

        optimized_paragraphs = []
        init_scores = []
        final_scores = []
        sim_scores = []

        for idx, para in enumerate(paragraphs):
            para_restored = parser.restore_tex(para, lock_dict)
            init_ai = self._predict_sentence_ai_prob(para_restored)
            init_scores.append(init_ai)

            # Run sentence/line-level closed-loop on paragraph
            if "__TEX_LOCK_ITEM_TAG_" in para or "\n" in para:
                sub_lines = [line.strip() for line in para.split("\n") if line.strip()]
                sentences = []
                for line in sub_lines:
                    sentences.extend(segment_sentences(line))
            else:
                sentences = segment_sentences(para)

            opt_sents = []
            sent_sims = []
            sent_ais = []

            for sent_locked in sentences:
                sent_restored = parser.restore_tex(sent_locked, lock_dict)
                sent_clean = sent_locked.strip()

                # Bypass candidate generation for pure locked blocks (math, preamble, bibitem, tables)
                if sent_clean.startswith("__TEX_LOCK_") and sent_clean.endswith("__"):
                    opt_sents.append(sent_restored)
                    sent_sims.append(1.0)
                    sent_ais.append(self._predict_sentence_ai_prob(sent_restored))
                    continue

                sent_ai_orig = self._predict_sentence_ai_prob(sent_restored)

                # Adaptive Early Exit: if sentence is already classified as natural human (< 15% AI score)
                if sent_ai_orig < 0.15:
                    opt_sents.append(sent_restored)
                    sent_sims.append(1.0)
                    sent_ais.append(sent_ai_orig)
                    continue

                cand_locked_list = self.humanizer.generate_neural_candidates(sentence=sent_locked, tone=tone)
                rule_locked = combined_rewrite(sent_locked, tone=tone)
                hybrid_locked = [combined_rewrite(nc, tone=tone) for nc in cand_locked_list]

                all_cand_locked = [rule_locked] + cand_locked_list + hybrid_locked + [sent_locked]

                valid_candidates = []
                for cand_locked in all_cand_locked:
                    cand_restored = parser.restore_tex(cand_locked, lock_dict)
                    is_valid, _ = self.guardrail.verify_academic_integrity(sent_restored, cand_restored, lock_dict)
                    if not is_valid:
                        continue
                    sim = self.guardrail.calculate_semantic_similarity(sent_restored, cand_restored)
                    if sim < min_semantic_sim:
                        continue
                    valid_candidates.append((cand_restored, sim))

                best_sent = sent_restored
                best_sent_ai = sent_ai_orig
                best_sent_sim = 1.0

                if valid_candidates:
                    cand_texts = [c[0] for c in valid_candidates]
                    cand_probs = self._predict_batch_ai_probs(cand_texts)

                    for (cand_restored, sim), cand_ai in zip(valid_candidates, cand_probs):
                        if cand_ai < best_sent_ai:
                            best_sent_ai = cand_ai
                            best_sent_sim = sim
                            best_sent = cand_restored

                opt_sents.append(best_sent)
                sent_sims.append(best_sent_sim)
                sent_ais.append(best_sent_ai)

            if "__TEX_LOCK_ITEM_TAG_" in para or "\n" in para:
                best_para = "\n".join(opt_sents)
            else:
                best_para = " ".join(opt_sents)

            best_sim = sum(sent_sims) / max(1, len(sent_sims))
            final_ai = sum(sent_ais) / max(1, len(sent_ais))

            optimized_paragraphs.append(best_para)
            final_scores.append(final_ai)
            sim_scores.append(best_sim)

            yield {
                "type": "tex_step",
                "paragraph_index": idx,
                "total_paragraphs": total_paras,
                "original_paragraph": para_restored,
                "humanized_paragraph": best_para,
                "initial_ai_score": round(init_ai, 4),
                "final_ai_score": round(final_ai, 4),
                "semantic_similarity": round(best_sim, 4)
            }

        # Reassemble document
        reassembled_locked = "\n\n".join(optimized_paragraphs)
        final_tex = parser.restore_tex(reassembled_locked, lock_dict)

        avg_init = sum(init_scores) / max(1, len(init_scores))
        avg_final = sum(final_scores) / max(1, len(final_scores))
        avg_sim = sum(sim_scores) / max(1, len(sim_scores))

        yield {
            "type": "tex_complete",
            "final_tex": final_tex,
            "avg_initial_ai_score": round(avg_init, 4),
            "avg_final_ai_score": round(avg_final, 4),
            "avg_semantic_similarity": round(avg_sim, 4),
            "total_paragraphs": total_paras
        }
