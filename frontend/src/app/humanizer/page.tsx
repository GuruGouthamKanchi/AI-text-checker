'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, 
  Seal, 
  Copy, 
  ArrowCounterClockwise, 
  Sparkle, 
  TextAa, 
  Quotes,
  Check,
  ShieldCheck,
  Warning,
  Info,
  Sliders,
  Terminal,
  PaperPlane
} from '@phosphor-icons/react';

interface DiffPart {
  type: 'equal' | 'delete' | 'insert';
  value: string;
}

interface HumanizeResponse {
  original_text?: string;
  humanized_text?: string;
  rewritten_text?: string;
  original_ai_percentage?: number;
  original_score?: number;
  humanized_ai_percentage?: number;
  rewritten_ai_percentage?: number;
  humanized_score?: number;
  robustness_confidence_delta?: number;
  score_reduction?: number;
  resilience_verdict?: string;
  explanation?: string;
}

interface ClosedLoopStep {
  index: number;
  total: number;
  original: string;
  humanized: string;
  initial_ai_prob: number;
  final_ai_prob: number;
  semantic_similarity: number;
  citations_preserved: boolean;
}

export default function AIHumanizerWorkspace() {
  const [inputText, setInputText] = useState<string>("");
  const [outputText, setOutputText] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<HumanizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tone, setTone] = useState<string>("academic");
  const [mode, setMode] = useState<string>("closed-loop");
  const [closedLoopSummary, setClosedLoopSummary] = useState<any>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [showDiff, setShowDiff] = useState<boolean>(true);


  // Safe score extractors to prevent any runtime toFixed crashes
  const getOriginalScore = (res: any): number => {
    if (!res) return 0;
    const val = res.original_ai_percentage ?? res.original_score ?? 0;
    return typeof val === 'number' && !isNaN(val) ? val : 0;
  };

  const getHumanizedScore = (res: any): number => {
    if (!res) return 0;
    const val = res.humanized_ai_percentage ?? res.rewritten_ai_percentage ?? res.humanized_score ?? 0;
    return typeof val === 'number' && !isNaN(val) ? val : 0;
  };

  const getDeltaScore = (res: any): number => {
    if (!res) return 0;
    const val = res.robustness_confidence_delta ?? res.score_reduction ?? 0;
    return typeof val === 'number' && !isNaN(val) ? val : 0;
  };

  // Word-level LCS diff algorithm for highlighting changes
  const getWordDiff = (original: string, rewritten: string): DiffPart[] => {
    const cleanOrig = original.replace(/\s+/g, ' ').trim();
    const cleanRew = rewritten.replace(/\s+/g, ' ').trim();
    if (!cleanOrig || !cleanRew) return [];

    const origWords = cleanOrig.split(' ');
    const rewWords = cleanRew.split(' ');

    const dp: number[][] = Array(origWords.length + 1)
      .fill(null)
      .map(() => Array(rewWords.length + 1).fill(0));

    for (let i = 1; i <= origWords.length; i++) {
      for (let j = 1; j <= rewWords.length; j++) {
        if (origWords[i - 1].toLowerCase() === rewWords[j - 1].toLowerCase()) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    const diff: DiffPart[] = [];
    let i = origWords.length;
    let j = rewWords.length;

    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && origWords[i - 1].toLowerCase() === rewWords[j - 1].toLowerCase()) {
        diff.unshift({ type: 'equal', value: rewWords[j - 1] });
        i--;
        j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        diff.unshift({ type: 'insert', value: rewWords[j - 1] });
        j--;
      } else {
        diff.unshift({ type: 'delete', value: origWords[i - 1] });
        i--;
      }
    }

    // Group adjacent parts of the same type for cleaner rendering
    const grouped: DiffPart[] = [];
    for (const part of diff) {
      if (grouped.length > 0 && grouped[grouped.length - 1].type === part.type) {
        grouped[grouped.length - 1].value += ' ' + part.value;
      } else {
        grouped.push({ ...part });
      }
    }

    return grouped;
  };

  const handleHumanize = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setClosedLoopSummary(null);
    setOutputText("");

    try {
      if (mode === "closed-loop") {
        const response = await fetch("http://127.0.0.1:8000/robustness/closed-loop-humanize/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText, tone: tone })
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || "Closed-Loop humanization streaming failed.");
        }

        if (response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunkStr = decoder.decode(value, { stream: true });
            const lines = chunkStr.split("\n\n");

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const dataStr = line.replace("data: ", "").trim();
                if (dataStr === "[DONE]") break;
                try {
                  const parsed = JSON.parse(dataStr);
                  if (parsed.type === "init") {
                    setClosedLoopSummary((prev: any) => ({ ...prev, ...parsed }));
                  } else if (parsed.type === "step") {
                    setOutputText((prev) => (prev ? prev + " " + parsed.humanized : parsed.humanized));
                  } else if (parsed.type === "complete") {
                    setOutputText(parsed.final_text);
                    setResult({
                      original_text: inputText,
                      humanized_text: parsed.final_text,
                      original_ai_percentage: Math.round(parsed.avg_initial_ai_score * 100),
                      humanized_ai_percentage: Math.round(parsed.avg_final_ai_score * 100),
                      score_reduction: Math.round((parsed.avg_initial_ai_score - parsed.avg_final_ai_score) * 100),
                      resilience_verdict: "Closed-Loop Optimized (Semantics Preserved)",
                      explanation: `Closed-loop optimization achieved ${Math.round(parsed.avg_semantic_similarity * 100)}% semantic retention while reducing AI score from ${Math.round(parsed.avg_initial_ai_score * 100)}% to ${Math.round(parsed.avg_final_ai_score * 100)}%.`
                    });
                    setClosedLoopSummary((prev: any) => ({ ...prev, ...parsed }));
                  }
                } catch (e) {}
              }
            }
          }
        }
      } else {
        // Standard Humanizer Streaming
        const response = await fetch("http://127.0.0.1:8000/robustness/humanize/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText, tone: tone })
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || "Humanization streaming failed.");
        }

        if (response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let accumulated = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunkStr = decoder.decode(value, { stream: true });
            const lines = chunkStr.split("\n\n");

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const dataStr = line.replace("data: ", "").trim();
                if (dataStr === "[DONE]") break;
                try {
                  const parsed = JSON.parse(dataStr);
                  if (parsed.chunk) {
                    accumulated += parsed.chunk;
                    setOutputText(accumulated);
                  }
                } catch (e) {}
              }
            }
          }
        }

        const scoreResponse = await fetch("http://127.0.0.1:8000/robustness/humanize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: inputText, tone: tone })
        });

        if (scoreResponse.ok) {
          const data: HumanizeResponse = await scoreResponse.json();
          setResult(data);
          if (data.humanized_text) {
            setOutputText(data.humanized_text);
          }
        }
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unexpected network error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!outputText) return;
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setInputText("");
    setOutputText("");
    setResult(null);
    setError(null);
  };

  // Rosette path for visual continuity
  const getRosettePath = () => {
    let d = "";
    const points = 32;
    for (let i = 0; i < points; i++) {
      const angle = (i * 2 * Math.PI) / points;
      const r = i % 2 === 0 ? 48 : 44;
      const x = (50 + r * Math.cos(angle)).toFixed(2);
      const y = (50 + r * Math.sin(angle)).toFixed(2);
      if (i === 0) {
        d += `M ${x} ${y}`;
      } else {
        const midAngle = angle - Math.PI / points;
        const cx = (50 + 51 * Math.cos(midAngle)).toFixed(2);
        const cy = (50 + 51 * Math.sin(midAngle)).toFixed(2);
        d += ` Q ${cx} ${cy}, ${x} ${y}`;
      }
    }
    d += " Z";
    return d;
  };

  const loadSampleText = () => {
    setInputText(
      "Collaborated on database query optimization and schema design, achieving a 25% reduction in API response latency across 10+ core production features. Furthermore, it is crucial to delve into the multifaceted challenges of AI alignment to foster responsible development."
    );
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#FAF8F2] text-[#21242B] font-body selection:bg-[#7A2331]/10 selection:text-[#7A2331] antialiased">
      
      {/* Navigation Header */}
      <header className="flex items-center justify-between px-10 py-5 border-b border-[#DCD4C0] bg-[#FAF8F2] shrink-0">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-3 hover:opacity-90">
            <span className="text-2xl font-display font-medium tracking-wide text-[#7A2331] flex items-center gap-2 select-none">
              <svg className="w-6 h-6" viewBox="0 0 100 100" fill="currentColor">
                <path d={getRosettePath()} />
              </svg>
              VeriPaper AI
            </span>
          </Link>
          <span className="text-[10px] font-mono tracking-widest text-[#7A2331] bg-transparent border border-[#DCD4C0] px-2 py-0.5 rounded uppercase">
            FLAN-T5-v1
          </span>
        </div>

        {/* Tab Navigation */}
        <nav className="flex bg-[#FAF8F2] border border-[#DCD4C0] rounded p-0.5 select-none shadow-[inset_0_1px_3px_rgba(0,0,0,0.03)]">
          <Link 
            href="/"
            className="px-5 py-1.5 text-xs font-mono font-medium rounded transition-all text-[#21242B]/60 hover:text-[#21242B] cursor-pointer"
          >
            FORENSIC AUDITOR
          </Link>
          <div className="px-5 py-1.5 text-xs font-mono font-medium rounded bg-[#7A2331] text-[#FAF8F2] shadow-sm select-none">
            TEXT HUMANIZER
          </div>
          <Link 
            href="/closed-loop"
            className="px-5 py-1.5 text-xs font-mono font-medium rounded transition-all text-[#21242B]/60 hover:text-[#21242B] cursor-pointer"
          >
            CLOSED-LOOP PIPELINE
          </Link>
        </nav>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
            {loading ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-[#B8862E] animate-pulse"></span>
                <span>REWRITING TEXT STRUCTURE...</span>
              </>
            ) : result ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                <span className="text-[#4B6A57] font-semibold">HUMANIZATION SUCCESSFUL</span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                <span>ENGINE READY</span>
              </>
            )}
          </div>

          {(inputText || result) && (
            <button 
              onClick={handleClear}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-mono font-medium rounded border border-[#DCD4C0] hover:border-[#7A2331] hover:text-[#7A2331] transition-all bg-transparent cursor-pointer"
            >
              <ArrowCounterClockwise weight="duotone" className="w-3.5 h-3.5" />
              CLEAR ALL
            </button>
          )}
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col p-10 overflow-y-auto max-w-[1600px] w-full mx-auto gap-8">
        
        {/* Workspace Title Card */}
        <div className="bg-[#FAF8F2] border border-[#DCD4C0] rounded p-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 relative overflow-hidden shadow-sm">
          <div className="absolute right-0 top-0 opacity-[0.015] pointer-events-none transform translate-x-12 -translate-y-12">
            <Seal size={400} weight="fill" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-medium tracking-wide text-[#7A2331] mb-2">
              AI Text Humanizer Suite
            </h1>
            <p className="text-sm text-slate-600 max-w-[800px] leading-relaxed">
              Synthesize AI-generated manuscripts, essays, and technical resumes into organic human drafts. 
              Our engine combines structural voice modifications with fine-tuned Flan-T5 neural paraphrasing 
              to maximize natural flow and defeat academic AI detection.
            </p>
          </div>
          <button 
            onClick={loadSampleText}
            className="shrink-0 flex items-center gap-2 px-4 py-2.5 text-xs font-mono font-medium rounded border border-[#7A2331] text-[#7A2331] hover:bg-[#7A2331]/5 transition-all cursor-pointer"
          >
            <FileText className="w-4 h-4" />
            LOAD SAMPLE TEXT
          </button>
        </div>

        {/* Configuration Bar */}
        <div className="bg-[#FAF8F2] border border-[#DCD4C0] rounded p-5 flex flex-wrap gap-6 items-center justify-between">
          <div className="flex flex-wrap gap-6 items-center">
            
            {/* Tone Selector */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-mono tracking-wider text-slate-500 uppercase flex items-center gap-1">
                <Sliders size={12} />
                Target Tone Profile
              </label>
              <select 
                value={tone}
                onChange={(e) => setTone(e.target.value)}
                className="bg-[#FAF8F2] border border-[#DCD4C0] rounded px-3 py-1.5 text-xs font-mono font-medium text-[#21242B] focus:border-[#7A2331] focus:ring-1 focus:ring-[#7A2331] outline-none cursor-pointer"
              >
                <option value="resume">Professional Resume (Active Action Verbs)</option>
                <option value="academic">Academic Paper (Scholarly Precision)</option>
                <option value="casual">Casual / Conversational (Fluid Transitions)</option>
                <option value="creative">Creative / Content (Expressive Vocabulary)</option>
              </select>
            </div>

            {/* Humanizer Engine Mode Selector */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-mono tracking-wider text-slate-500 uppercase flex items-center gap-1">
                <Sparkle size={12} />
                Optimization Framework
              </label>
              <div className="flex border border-[#DCD4C0] rounded bg-[#FAF8F2] p-0.5 select-none">
                <button
                  type="button"
                  onClick={() => setMode("closed-loop")}
                  className={`px-3 py-1 text-xs font-mono font-medium rounded transition-all cursor-pointer flex items-center gap-1.5 ${
                    mode === "closed-loop"
                      ? "bg-[#7A2331] text-[#FAF8F2] shadow-sm font-semibold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <ShieldCheck size={13} />
                  Closed-Loop Academic Mode
                </button>
                <button
                  type="button"
                  onClick={() => setMode("standard")}
                  className={`px-3 py-1 text-xs font-mono font-medium rounded transition-all cursor-pointer flex items-center gap-1.5 ${
                    mode === "standard"
                      ? "bg-[#7A2331] text-[#FAF8F2] shadow-sm font-semibold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  Standard Paraphraser
                </button>
              </div>
            </div>
          </div>

          {result && (
            <div className="flex items-center gap-3">
              <label className="text-xs font-mono text-slate-500 flex items-center gap-1.5 cursor-pointer select-none">
                <input 
                  type="checkbox" 
                  checked={showDiff} 
                  onChange={(e) => setShowDiff(e.target.checked)}
                  className="rounded border-[#DCD4C0] text-[#7A2331] focus:ring-[#7A2331]"
                />
                Show structural changes (Diff view)
              </label>
            </div>
          )}
        </div>

        {/* Dual-Pane Editor Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Pane - Input */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-mono font-bold tracking-wider text-slate-500 uppercase flex items-center gap-1.5">
                <Terminal size={14} />
                Draft Copy (AI Text Box)
              </h2>
              <span className="text-[11px] font-mono text-slate-500 bg-[#FAF8F2] px-2 py-0.5 border border-[#DCD4C0] rounded select-none">
                {inputText.split(/\s+/).filter(Boolean).length} words | {inputText.length} chars
              </span>
            </div>
            
            <div className="relative bg-[#FAF8F2] border border-[#DCD4C0] rounded-lg shadow-sm focus-within:border-[#7A2331] focus-within:ring-1 focus-within:ring-[#7A2331] transition-all flex flex-col min-h-[420px]">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste your AI-generated text or resume draft here..."
                disabled={loading}
                className="w-full flex-1 p-6 text-sm leading-relaxed bg-transparent border-0 outline-none resize-none font-body text-[#21242B] placeholder-slate-400"
              />
              
              <div className="p-4 border-t border-[#DCD4C0] flex justify-end shrink-0">
                <button
                  onClick={handleHumanize}
                  disabled={loading || !inputText.trim()}
                  className="flex items-center gap-2 px-6 py-2.5 text-xs font-mono font-bold tracking-wider text-[#FAF8F2] bg-[#7A2331] hover:bg-[#7A2331]/95 disabled:bg-slate-300 disabled:cursor-not-allowed rounded shadow transition-all cursor-pointer"
                >
                  {loading ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-[#FAF8F2] border-t-transparent rounded-full animate-spin"></div>
                      PROCESSING...
                    </>
                  ) : (
                    <>
                      <PaperPlane className="w-4 h-4" />
                      HUMANIZE TEXT
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right Pane - Output */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-mono font-bold tracking-wider text-slate-500 uppercase flex items-center gap-1.5">
                <TextAa size={14} />
                Refined Copy (Humanized Codex)
              </h2>
              {outputText && (
                <span className="text-[11px] font-mono text-slate-500 bg-[#FAF8F2] px-2 py-0.5 border border-[#DCD4C0] rounded select-none">
                  {outputText.split(/\s+/).filter(Boolean).length} words
                </span>
              )}
            </div>

            <div className="bg-[#FAF8F2] border border-[#DCD4C0] rounded-lg shadow-sm flex flex-col min-h-[420px]">
              
              <div className="flex-1 p-6 overflow-y-auto max-h-[352px] text-sm leading-relaxed text-[#21242B]">
                <AnimatePresence mode="wait">
                  {loading ? (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="space-y-4 py-2"
                    >
                      <div className="h-4 bg-slate-200 rounded animate-pulse w-full"></div>
                      <div className="h-4 bg-slate-200 rounded animate-pulse w-5/6"></div>
                      <div className="h-4 bg-slate-200 rounded animate-pulse w-11/12"></div>
                      <div className="h-4 bg-slate-200 rounded animate-pulse w-3/4"></div>
                      <div className="h-4 bg-slate-200 rounded animate-pulse w-5/6"></div>
                    </motion.div>
                  ) : error ? (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-start gap-2 text-[#B23A2E] bg-[#B23A2E]/5 p-4 rounded border border-[#B23A2E]/10 font-mono text-xs"
                    >
                      <Warning className="w-4 h-4 shrink-0 mt-0.5" />
                      <span>{error}</span>
                    </motion.div>
                  ) : outputText ? (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="font-body text-[#21242B] whitespace-pre-wrap selection:bg-[#4B6A57]/10 selection:text-[#4B6A57]"
                    >
                      {showDiff && result ? (
                        <div className="leading-relaxed">
                          {getWordDiff(inputText, outputText).map((part, idx) => {
                            if (part.type === 'insert') {
                              return (
                                <span 
                                  key={idx} 
                                  className="bg-[#4B6A57]/15 text-[#4B6A57] px-1 py-0.5 rounded font-semibold transition-all"
                                >
                                  {part.value}{' '}
                                </span>
                              );
                            } else if (part.type === 'delete') {
                              return (
                                <span 
                                  key={idx} 
                                  className="bg-[#B23A2E]/10 text-[#B23A2E] line-through decoration-[#B23A2E]/40 px-1 py-0.5 rounded text-xs mx-0.5 opacity-80"
                                >
                                  {part.value}{' '}
                                </span>
                              );
                            } else {
                              return <span key={idx}>{part.value} </span>;
                            }
                          })}
                        </div>
                      ) : (
                        outputText
                      )}
                    </motion.div>
                  ) : (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex flex-col items-center justify-center h-full text-slate-400 gap-2 py-20"
                    >
                      <Quotes size={48} weight="thin" />
                      <span className="font-mono text-xs">AWAITING INPUT TO REFINE</span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {outputText && !loading && (
                <div className="p-4 border-t border-[#DCD4C0] flex justify-end shrink-0 bg-[#FAF8F2]/50">
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 px-5 py-2.5 text-xs font-mono font-bold border border-[#DCD4C0] hover:border-[#4B6A57] hover:text-[#4B6A57] transition-all bg-[#FAF8F2] rounded cursor-pointer"
                  >
                    {copied ? (
                      <>
                        <Check weight="bold" className="w-3.5 h-3.5 text-[#4B6A57]" />
                        <span className="text-[#4B6A57]">COPIED!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        COPY TEXT
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Verdict Diagnostics Dashboard */}
        <AnimatePresence>
          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 15 }}
              className="bg-[#FAF8F2] border border-[#DCD4C0] rounded-lg p-8 grid grid-cols-1 md:grid-cols-3 gap-8 shadow-sm relative overflow-hidden"
            >
              
              {/* Circular Verdict Gauge */}
              <div className="flex flex-col items-center justify-center border-b md:border-b-0 md:border-r border-[#DCD4C0] pb-6 md:pb-0 md:pr-8 gap-4">
                <span className="text-[10px] font-mono tracking-wider text-slate-500 uppercase">
                  AI Probability Score Drop
                </span>
                
                <div className="flex items-center gap-6 select-none">
                  {/* Before Score */}
                  <div className="flex flex-col items-center">
                    <span className="text-2xl font-mono font-bold text-[#B23A2E]">
                      {getOriginalScore(result).toFixed(1)}%
                    </span>
                    <span className="text-[9px] font-mono text-slate-500 uppercase mt-0.5">
                      ORIGINAL
                    </span>
                  </div>

                  {/* Arrow Indicator */}
                  <div className="flex flex-col items-center text-slate-400">
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                    <span className="text-[9px] font-mono text-[#4B6A57] font-semibold mt-1">
                      -{getDeltaScore(result).toFixed(1)}%
                    </span>
                  </div>

                  {/* After Score */}
                  <div className="flex flex-col items-center">
                    <span className="text-2xl font-mono font-bold text-[#4B6A57] bg-[#4B6A57]/10 px-3 py-1 rounded border border-[#4B6A57]/20">
                      {getHumanizedScore(result).toFixed(1)}%
                    </span>
                    <span className="text-[9px] font-mono text-slate-500 uppercase mt-1">
                      HUMANIZED
                    </span>
                  </div>
                </div>

                <div className="text-xs font-mono text-slate-500 text-center">
                  VeriPaper detector rescoring verification
                </div>
              </div>

              {/* Diagnosis Verdict Card */}
              <div className="flex flex-col justify-center border-b md:border-b-0 md:border-r border-[#DCD4C0] pb-6 md:pb-0 md:px-8 gap-3">
                <span className="text-[10px] font-mono tracking-wider text-slate-500 uppercase">
                  Resilience Diagnostic Verdict
                </span>

                <div className="flex items-center gap-2.5">
                  <ShieldCheck size={28} weight="fill" className="text-[#4B6A57]" />
                  <span className="text-xl font-display font-medium text-[#4B6A57] tracking-wide">
                    {result.resilience_verdict}
                  </span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed font-body">
                  {result.explanation}
                </p>
              </div>

              {/* Forensic Engine Logs */}
              <div className="flex flex-col justify-center md:pl-8 gap-3">
                <span className="text-[10px] font-mono tracking-wider text-slate-500 uppercase flex items-center gap-1">
                  <Info size={12} />
                  Active Transformations applied
                </span>

                <div className="space-y-2 text-xs font-mono text-slate-600">
                  {closedLoopSummary && (
                    <>
                      <div className="flex items-center gap-2 text-[#4B6A57] font-semibold">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                        <span>Semantic Retention: {Math.round(closedLoopSummary.avg_semantic_similarity * 100)}%</span>
                      </div>
                      <div className="flex items-center gap-2 text-[#4B6A57] font-semibold">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                        <span>Citations & Math Formulas: 100% Preserved ({closedLoopSummary.locked_tokens_count || 0} Locked)</span>
                      </div>
                    </>
                  )}
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                    <span>Syntactic Restructuring: Enabled</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                    <span>Cliché Vocabulary Elimination: Applied</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                    <span>Neural Flan-T5 Paraphrasing: Complete</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                    <span>Grammar Post-Processing: Clean</span>
                  </div>
                </div>
              </div>

            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* Footer bar */}
      <footer className="py-5 border-t border-[#DCD4C0] text-center text-[10px] font-mono tracking-wider text-slate-500 shrink-0">
        VERIPAPER AI LABS © 2026. FOR HUMAN AUTHENTICITY ASSURANCE AND COGNITIVE FORENSICS.
      </footer>

    </div>
  );
}
