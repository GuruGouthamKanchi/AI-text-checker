'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { ThinkingOrb } from '@/components/ThinkingOrb';
import { TubesCursor } from '@/components/TubesCursor';
import { 
  FileText, 
  ArrowCounterClockwise, 
  Sparkle, 
  TextAa, 
  Check, 
  ShieldCheck, 
  Warning, 
  Info, 
  DownloadSimple, 
  UploadSimple, 
  Code,
  PaperPlane,
  Copy,
  CaretRight,
  Lightning,
  Trash,
  ArrowLeft
} from '@phosphor-icons/react';

export default function ClosedLoopWorkspace() {
  const [tab, setTab] = useState<'text' | 'tex'>('tex');
  const [inputText, setInputText] = useState<string>('');
  const [outputText, setOutputText] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [tone, setTone] = useState<string>('academic');
  const [copied, setCopied] = useState<boolean>(false);

  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = sessionStorage.getItem('humanize_input');
      if (stored) {
        setInputText(stored);
        sessionStorage.removeItem('humanize_input');
      } else {
        const urlParams = new URLSearchParams(window.location.search);
        const textParam = urlParams.get('text');
        if (textParam) setInputText(textParam);
      }
    }
  }, []);

  // Closed-Loop Stream States
  const [summary, setSummary] = useState<any>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(0);

  const handleStartClosedLoop = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setSummary(null);
    setOutputText('');
    setCurrentStepIndex(0);
    setTotalSteps(0);

    const host = typeof window !== 'undefined' && window.location.hostname ? window.location.hostname : '127.0.0.1';
    const endpoint =
      tab === 'tex'
        ? `http://${host}:8000/robustness/humanize-tex/stream`
        : `http://${host}:8000/robustness/closed-loop-humanize/stream`;

    try {
      let response: Response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: inputText, tone: tone })
        });
      } catch (e) {
        const fallbackEndpoint = tab === 'tex'
          ? 'http://127.0.0.1:8000/robustness/humanize-tex/stream'
          : 'http://127.0.0.1:8000/robustness/closed-loop-humanize/stream';
        response = await fetch(fallbackEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: inputText, tone: tone })
        });
      }

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || 'Closed-Loop optimization stream failed.');
      }

      if (response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunkStr = decoder.decode(value, { stream: true });
          const lines = chunkStr.split('\n\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.replace('data: ', '').trim();
              if (dataStr === '[DONE]') break;

              try {
                const parsed = JSON.parse(dataStr);
                if (parsed.type === 'init' || parsed.type === 'tex_init') {
                  setTotalSteps(parsed.total_sentences || parsed.total_paragraphs || 0);
                  setSummary((prev: any) => ({ ...prev, ...parsed }));
                } else if (parsed.type === 'step' || parsed.type === 'tex_step') {
                  setCurrentStepIndex((prev) => prev + 1);
                  const textChunk = parsed.humanized || parsed.humanized_paragraph || '';
                  setOutputText((prev) => (prev ? prev + '\n\n' + textChunk : textChunk));
                } else if (parsed.type === 'complete' || parsed.type === 'tex_complete') {
                  const finalText = parsed.final_text || parsed.final_tex || '';
                  setOutputText(finalText);
                  setSummary((prev: any) => ({ ...prev, ...parsed }));
                }
              } catch (e) {}
            }
          }
        }
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An unexpected network error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadTex = () => {
    const textToDownload = outputText || inputText;
    if (!textToDownload.trim()) return;

    const blob = new Blob([textToDownload], { type: 'application/x-tex' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'refined_academic_paper.tex';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setInputText(content);
      if (file.name.endsWith('.tex')) setTab('tex');
    };
    reader.readAsText(file);
  };

  const handleCopy = () => {
    if (!outputText) return;
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setInputText('');
    setOutputText('');
    setSummary(null);
    setError(null);
    setCurrentStepIndex(0);
    setTotalSteps(0);
  };

  return (
    <div className="w-full max-w-[1920px] min-h-screen mx-auto bg-[#090d16] text-slate-100 font-sans antialiased selection:bg-indigo-500/30 relative">
      <TubesCursor title="" subtitle="" caption="" className="fixed inset-0 z-0 pointer-events-none opacity-20" />
      
      {/* Navigation Header */}
      <header className="w-full h-16 px-6 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <Link href="/" className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-all">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 via-teal-600 to-indigo-500 p-0.5 shadow-lg shadow-emerald-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Sparkle className="w-5 h-5 text-emerald-400" />
            </div>
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-100 tracking-tight flex items-center gap-2">
              Closed-Loop Dual-Agent Optimizer <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800/60 font-semibold font-mono">ModernBERT Active</span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => setTab('tex')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${tab === 'tex' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
            >
              <Code className="w-3.5 h-3.5" />
              LaTeX Mode (.tex)
            </button>
            <button
              onClick={() => setTab('text')}
              className={`px-3 py-1.5 rounded-lg font-semibold transition-all flex items-center gap-1.5 ${tab === 'text' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
            >
              <TextAa className="w-3.5 h-3.5" />
              Plain Text Mode
            </button>
          </div>

          <label className="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-xl text-xs font-semibold text-slate-300 hover:bg-slate-800 cursor-pointer transition-all flex items-center gap-1.5">
            <UploadSimple className="w-4 h-4 text-indigo-400" />
            Upload File
            <input type="file" onChange={handleFileUpload} accept=".tex,.txt" className="hidden" />
          </label>

          <button
            onClick={handleClear}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-red-400 hover:bg-red-950/40 transition-all"
            title="Clear Workspace"
          >
            <Trash className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Full-Width Dual Canvas Split (50% Input / 50% Streamed Output) */}
      <main className="w-full p-4 flex gap-4 min-h-[calc(100vh-4rem)]">
        
        {/* Left Side: Input Document Area */}
        <section className="flex-1 bg-slate-950/70 border border-slate-800/80 rounded-2xl flex flex-col overflow-hidden shadow-2xl backdrop-blur-sm">
          <div className="px-5 py-3.5 bg-slate-900/60 border-b border-slate-800/80 flex items-center justify-between">
            <span className="font-semibold text-sm text-slate-200 flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-400" />
              Original Manuscript Source Input ({tab.toUpperCase()})
            </span>
            <span className="text-xs text-slate-400 font-mono">
              {inputText.length} characters
            </span>
          </div>

          <div className="flex-1 p-4 flex flex-col">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={tab === 'tex' ? "\\documentclass{article}\n\\begin{document}\nPaste your LaTeX document here...\n\\end{document}" : "Paste raw academic text here..."}
              className="w-full flex-1 p-4 bg-slate-950 text-slate-200 font-mono text-xs rounded-xl border border-slate-800 focus:outline-none focus:border-indigo-500/60 resize-none leading-relaxed"
            />
            
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400 font-medium">Academic Tone:</span>
                <select 
                  value={tone} 
                  onChange={(e) => setTone(e.target.value)}
                  className="bg-slate-900 text-slate-200 text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-800 focus:outline-none focus:border-indigo-500"
                >
                  <option value="academic">Academic Formal (Default)</option>
                  <option value="formal">Journal Formal</option>
                  <option value="essay">Thesis Essay</option>
                </select>
              </div>

              <button
                onClick={handleStartClosedLoop}
                disabled={loading || !inputText.trim()}
                className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-indigo-600 hover:from-emerald-500 hover:to-indigo-500 text-white font-bold text-xs tracking-wide transition-all shadow-lg shadow-emerald-600/20 flex items-center gap-2 disabled:opacity-50 cursor-pointer"
              >
                {loading ? <Lightning className="w-4 h-4 animate-spin" /> : <Sparkle className="w-4 h-4" />}
                {loading ? 'Optimizing Sentences...' : 'Start Closed-Loop Optimization'}
              </button>
            </div>
          </div>
        </section>

        {/* Right Side: Streamed Repaired Output Area */}
        <section className="flex-1 bg-slate-950/70 border border-slate-800/80 rounded-2xl flex flex-col overflow-hidden shadow-2xl backdrop-blur-sm">
          <div className="px-5 py-3.5 bg-slate-900/60 border-b border-slate-800/80 flex items-center justify-between">
            <span className="font-semibold text-sm text-slate-200 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Streamed Humanized & Repaired Output
            </span>

            {summary && (
              <div className="flex items-center gap-3 text-xs font-mono">
                <span className="text-slate-400">Baseline AI: <strong className="text-red-400">{Math.round((summary.initial_ai_ratio || 0) * 100)}%</strong></span>
                <CaretRight className="text-slate-600 w-3 h-3" />
                <span className="text-slate-400">Final AI: <strong className="text-emerald-400">{Math.round((summary.final_ai_ratio || 0) * 100)}%</strong></span>
              </div>
            )}
          </div>

          <div className="flex-1 p-4 flex flex-col relative">
            {loading && (
              <div className="w-full py-2 px-4 bg-emerald-950/50 border border-emerald-800/60 rounded-xl mb-3 flex items-center justify-between text-xs text-emerald-300 font-mono">
                <div className="flex items-center gap-2">
                  <ThinkingOrb state="analyzing" size={28} />
                  <span>Closed-Loop SSE Stream Step {currentStepIndex} of {totalSteps || '?'}...</span>
                </div>
                <div className="w-24 bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                  <div className="bg-emerald-500 h-full transition-all duration-300" style={{ width: `${totalSteps ? (currentStepIndex / totalSteps) * 100 : 50}%` }} />
                </div>
              </div>
            )}

            <textarea
              readOnly
              value={outputText}
              placeholder="Streamed humanized output will appear here in real-time step-by-step..."
              className="w-full flex-1 p-4 bg-slate-950 text-emerald-100/90 font-mono text-xs rounded-xl border border-slate-800 focus:outline-none leading-relaxed resize-none"
            />

            <div className="mt-4 flex items-center justify-between">
              <span className="text-xs text-slate-400">
                {summary?.converged ? <strong className="text-emerald-400 font-semibold">✓ Optimization Converged Below Target AI Threshold</strong> : 'Ready for export.'}
              </span>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  disabled={!outputText}
                  className="px-4 py-2 bg-slate-900 border border-slate-800 rounded-xl text-xs font-semibold text-slate-300 hover:bg-slate-800 disabled:opacity-40 transition-all flex items-center gap-1.5"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  {copied ? 'Copied!' : 'Copy Text'}
                </button>

                <button
                  onClick={handleDownloadTex}
                  disabled={!outputText && !inputText}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-bold transition-all shadow-md shadow-emerald-600/20 disabled:opacity-40 flex items-center gap-1.5 cursor-pointer"
                >
                  <DownloadSimple className="w-4 h-4" />
                  Download Repaired .tex
                </button>
              </div>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
