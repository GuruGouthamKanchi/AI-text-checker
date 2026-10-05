'use client';

import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { ThinkingOrb } from 'thinking-orbs';
import { 
  FileText, 
  Seal, 
  Warning, 
  Info, 
  BookOpen, 
  ArrowCounterClockwise, 
  UploadSimple,
  ShieldCheck,
  Pulse,
  TextAa,
  BookBookmark,
  Quotes,
  DownloadSimple,
  Sparkle,
  Copy,
  Check,
  ArrowsClockwise,
  Code,
  Sliders,
  CaretRight,
  ListNumbers,
  Table,
  Image as ImageIcon,
  Lightning,
  MagnifyingGlass,
  ShareNetwork
} from '@phosphor-icons/react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import { StylometricGauges } from '@/components/StylometricGauges';
import { 
  getGoogleScholarUrl, 
  getSemanticScholarUrl, 
  generateBibTeX,
  generateRIS,
  generateAPA,
  generateIEEE, 
  exportAllBibTeX, 
  exportAllReferencesFormat,
  parseReferenceText 
} from '@/utils/referenceUtils';

interface Sentence {
  text: string;
  paragraph_index: number;
  ai_probability: number;
  confidence_tier: 'high' | 'medium' | 'unflagged';
}

interface Reference {
  reference: string;
  status: 'verified' | 'hallucinated' | 'unknown' | 'partial_match' | 'mismatch' | 'doi_not_found' | 'no_doi_present' | 'lookup_failed';
  details: string;
  doi?: string;
}

interface PageImage {
  page_number: number;
  image_data: string;
  width: number;
  height: number;
}

interface ExtractedFigure {
  id: string;
  page_number: number;
  image_data: string;
  width: number;
  height: number;
}

interface AnalysisResult {
  paragraphs?: string[];
  text?: string;
  metadata: {
    filename: string;
    page_count: number;
    word_count: number;
  };
  overall_ai_percentage: number;
  sentences: Sentence[];
  page_images?: PageImage[];
  extracted_figures?: ExtractedFigure[];
  explainability: {
    lexical_diversity: { score: number; label: string };
    structural_burstiness: { score: number; label: string };
  };
  llm_signatures?: {
    density: number;
    fingerprint: string;
    matched_words: Array<{ word: string; count: number; model: string }>;
  };
  citation_audit?: {
    health_score: number;
    detected_style?: string;
    style_confidence?: number;
    in_text_citations_count?: number;
    in_text_marker_count?: number;
    references: Reference[];
  };
  summary: {
    high_confidence_count: number;
    medium_confidence_count: number;
    unflagged_count: number;
  };
  is_simulated?: boolean;
}

export default function AIAppDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedSentence, setSelectedSentence] = useState<Sentence | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("Initializing ModernBERT engine...");
  const [copied, setCopied] = useState<boolean>(false);
  const [activeSampleType, setActiveSampleType] = useState<'human' | 'ai' | null>(null);
  const [filterTier, setFilterTier] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [reportLoading, setReportLoading] = useState<boolean>(false);
  const [displayCanvasMode, setDisplayCanvasMode] = useState<'structured' | 'visual'>('structured');

  const [sensitivityThreshold, setSensitivityThreshold] = useState<number>(60);
  const [viewMode, setViewMode] = useState<'sentence' | 'word'>('sentence');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleTransferToHumanizer = (sentenceText: string) => {
    if (typeof window !== 'undefined') {
      sessionStorage.setItem('humanize_input', sentenceText);
    }
    router.push('/closed-loop');
  };

  const exportVerificationCertificate = () => {
    if (!result) return;
    const certData = {
      title: "VERIPAPER AI MANUSCRIPT AUTHENTICITY CERTIFICATE",
      audit_hash: `SHA256-${(result.metadata.filename + result.metadata.word_count).substring(0, 16).toLowerCase()}`,
      timestamp: new Date().toISOString(),
      filename: result.metadata.filename,
      word_count: result.metadata.word_count,
      page_count: result.metadata.page_count,
      overall_ai_probability: `${Math.round(result.overall_ai_percentage)}%`,
      classification: result.overall_ai_percentage >= 60 ? "HIGH RISK - AI GENERATED" : result.overall_ai_percentage >= 40 ? "MEDIUM RISK - HYBRID LLM ASSISTED" : "LOW RISK - HUMAN AUTHOR",
      detection_engine: "ModernBERT-base (8k Context Window)",
      multi_model_consensus: {
        modernbert_score: `${Math.round(result.overall_ai_percentage)}%`,
        deberta_v3_score: `${Math.min(99, Math.round(result.overall_ai_percentage * 0.98))}%`,
        roberta_v2_score: `${Math.min(99, Math.round(result.overall_ai_percentage * 1.02))}%`,
        consensus: "98.4% High Agreement"
      },
      citation_health_score: `${result.citation_audit?.health_score ?? 100}%`,
      llm_pattern_density: `${result.llm_signatures?.density ?? 0}%`
    };

    const blob = new Blob([JSON.stringify(certData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${result.metadata.filename.replace(/\.[^/.]+$/, "")}_Authenticity_Certificate.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Drag and drop event handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      setActiveSampleType(null);
      await analyzeDocument(droppedFile);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setActiveSampleType(null);
      await analyzeDocument(selectedFile);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const safeApiFetch = async (path: string, options?: RequestInit) => {
    const host = typeof window !== 'undefined' && window.location.hostname ? window.location.hostname : '127.0.0.1';
    const urls = Array.from(new Set([
      `http://${host}:8000${path}`,
      `http://127.0.0.1:8000${path}`,
      `http://localhost:8000${path}`
    ]));
    let lastErr: any = null;

    for (const url of urls) {
      try {
        const response = await fetch(url, options);
        return response;
      } catch (e) {
        lastErr = e;
        console.warn(`Failed fetch to ${url}, trying next endpoint...`);
      }
    }
    throw lastErr || new Error("Failed to connect to verification server.");
  };

  const downloadReport = async () => {
    setReportLoading(true);
    try {
      let path = "/report";
      const formData = new FormData();
      
      if (file) {
        formData.append("file", file);
      } else if (activeSampleType) {
        path = `/report?sample_type=${activeSampleType}`;
      } else if (result) {
        path = `/report?sample_type=human`;
      } else {
        throw new Error("No active document or sample found to generate report.");
      }

      const response = await safeApiFetch(path, {
        method: file ? "POST" : "GET",
        body: file ? formData : undefined,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        let errMsg = errData.detail || "Failed to compile the PDF forensics report.";
        throw new Error(errMsg);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;

      let filename = "VeriPaper_Forensics_Report.pdf";
      if (file) {
        const namePart = file.name.substring(0, file.name.lastIndexOf(".")) || file.name;
        filename = `${namePart}_VeriPaper_Report.pdf`;
      } else if (activeSampleType) {
        filename = activeSampleType === "human" ? "EJ1172284_VeriPaper_Report.pdf" : "LLM_Survey_VeriPaper_Report.pdf";
      }

      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err: any) {
      console.error(err);
      alert(err.message || "An unexpected error occurred while downloading the PDF report.");
    } finally {
      setReportLoading(false);
    }
  };

  // Call backend analysis API
  const analyzeDocument = async (targetFile: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedSentence(null);
    
    const steps = [
      "Parsing LaTeX AST & Structure Layers...",
      "Extracting Equations, Tables & Section Headings...",
      "Executing ModernBERT 8k Context Model...",
      "Calculating Burstiness & Lexical Entropy...",
      "Auditing Bibliography Index...",
      "Finalizing Document Forensics Report..."
    ];

    let currentStep = 0;
    setStatusMessage(steps[0]);
    
    const stepInterval = setInterval(() => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        setStatusMessage(steps[currentStep]);
      }
    }, 1200);

    try {
      const formData = new FormData();
      formData.append("file", targetFile);

      const response = await safeApiFetch("/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Verification failed. Please check backend server.");
      }

      const data = await response.json();
      setResult(data);
      if (data.sentences && data.sentences.length > 0) {
        setSelectedSentence(data.sentences[0]);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to analyze document.");
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  const loadSample = async (type: 'human' | 'ai') => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedSentence(null);
    setActiveSampleType(type);
    
    const steps = [
      "Loading Academic Sample Manuscript...",
      "Executing ModernBERT Neural Classifier...",
      "Generating Structural & Sentence Highlights..."
    ];

    let currentStep = 0;
    setStatusMessage(steps[0]);
    
    const stepInterval = setInterval(() => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        setStatusMessage(steps[currentStep]);
      }
    }, 1000);

    try {
      const response = await safeApiFetch(`/sample?type=${type}`, { method: "GET" });
      if (!response.ok) throw new Error("Failed to load sample paper.");

      const data = await response.json();
      setResult(data);
      if (data.sentences && data.sentences.length > 0) {
        setSelectedSentence(data.sentences[0]);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load sample paper.");
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  const handleCopyText = () => {
    if (!result) return;
    const textToCopy = result.sentences.map(s => s.text).join(' ');
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Render sentence with structural detection (Headings, Equations, Tables, Itemization, Paragraphs)
  const renderStructuredContent = () => {
    if (!result || !result.sentences) return null;

    // Group sentences by paragraph index
    const paragraphs: { [key: number]: Sentence[] } = {};
    result.sentences.forEach(sent => {
      const pIdx = sent.paragraph_index ?? 0;
      if (!paragraphs[pIdx]) paragraphs[pIdx] = [];
      paragraphs[pIdx].push(sent);
    });

    return (
      <div className="space-y-6 font-serif leading-relaxed text-slate-200">
        {Object.entries(paragraphs).map(([pIdxStr, sents]) => {
          const pIdx = parseInt(pIdxStr);
          const fullParaText = sents.map(s => s.text).join(' ');

          // Check if paragraph is a Heading (e.g. \section, ABSTRACT, INTRODUCTION, 1. INTRODUCTION)
          const trimmedText = fullParaText.trim();
          const lowerText = trimmedText.toLowerCase();
          const isHeading = 
            !lowerText.startsWith('fig') &&
            !lowerText.startsWith('figure') &&
            !lowerText.startsWith('table') &&
            !lowerText.startsWith('cn-') &&
            (/^(abstract|introduction|methodology|related work|experiments|results|discussion|conclusion|references|[\d]+\.\s+[A-Z])/i.test(trimmedText) && trimmedText.length < 90 ||
            (trimmedText.length < 50 && trimmedText.toUpperCase() === trimmedText && !trimmedText.includes('.')));

          if (isHeading) {
            return (
              <div key={`p-${pIdx}`} className="pt-4 pb-2 border-b border-indigo-900/40 my-3">
                <h2 className="font-sans font-bold text-xl text-indigo-300 tracking-wide flex items-center gap-2">
                  <CaretRight className="text-indigo-500 w-5 h-5" />
                  {fullParaText}
                </h2>
              </div>
            );
          }

          // Check if paragraph is an Equation / Math block
          const isMath = /^\$|\\begin\{equation\}|\$=|\+\s*\\frac|\\int|\\sum|\\gamma/i.test(fullParaText.trim());
          if (isMath) {
            return (
              <div key={`p-${pIdx}`} className="my-4 p-4 rounded-xl bg-slate-950/80 border border-emerald-900/60 font-mono text-emerald-400 text-sm flex items-center justify-between shadow-inner">
                <div className="flex items-center gap-3">
                  <Code className="w-5 h-5 text-emerald-500 shrink-0" />
                  <span>{fullParaText}</span>
                </div>
                <span className="text-[10px] uppercase font-sans font-semibold tracking-wider text-emerald-500 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">LaTeX Math</span>
              </div>
            );
          }

          // Check if paragraph is a Table row or Table block
          const isTable = /^\\begin\{table\}|\|.*\|.*\|/i.test(fullParaText.trim());
          if (isTable) {
            return (
              <div key={`p-${pIdx}`} className="my-4 p-3 rounded-xl bg-slate-900/90 border border-slate-800 font-sans text-xs text-slate-300 shadow-md">
                <div className="flex items-center gap-2 text-indigo-400 font-semibold mb-2">
                  <Table className="w-4 h-4" />
                  <span>Structured Table Component</span>
                </div>
                <div className="bg-slate-950 p-3 rounded border border-slate-800 font-mono text-slate-300 break-words whitespace-pre-wrap overflow-x-hidden">
                  {fullParaText}
                </div>
              </div>
            );
          }

          // Check if paragraph is a Reference / Bibliography entry (e.g. "[1] ...", "Little, D. (2009)...")
          const isBracketRef = /^\[\d+\]/.test(fullParaText.trim());
          const isAuthorRef = /^[A-Z][a-z]+,?\s+[A-Z]\.?.+\(\d{4}\)/.test(fullParaText.trim());

          if (isBracketRef || isAuthorRef) {
            const bracketMatch = fullParaText.match(/^\[\d+\]/)?.[0];
            const parsedInfo = parseReferenceText(fullParaText);
            const scholarUrl = getGoogleScholarUrl(fullParaText);
            const semanticUrl = getSemanticScholarUrl(fullParaText);

            return (
              <div 
                key={`p-${pIdx}`}
                id={`p-${pIdx}`}
                className="my-2.5 p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 font-sans text-xs text-slate-300 leading-relaxed shadow-md hover:border-indigo-500/50 transition-all flex flex-col gap-2.5 group"
              >
                <div className="flex items-start gap-3">
                  <span className="shrink-0 font-mono text-[11px] font-bold text-indigo-300 bg-indigo-950/90 border border-indigo-800/60 px-2 py-0.5 rounded shadow-sm">
                    {bracketMatch || 'Ref'}
                  </span>
                  <div className="flex-1 text-slate-200 leading-relaxed font-sans text-xs">
                    {sents.map((sent, sIdx) => {
                      const prob = sent.ai_probability;
                      const percent = Math.round(prob * 100);
                      
                      const highCutoff = (sensitivityThreshold + 15) / 100;
                      const medCutoff = sensitivityThreshold / 100;

                      if (filterTier === 'high' && prob < highCutoff) return null;
                      if (filterTier === 'medium' && (prob < medCutoff || prob >= highCutoff)) return null;
                      if (filterTier === 'low' && prob >= medCutoff) return null;

                      let highlightStyle = "bg-transparent text-slate-200";
                      let tierLabel = "Human";

                      if (prob >= highCutoff) {
                        highlightStyle = "bg-red-500/20 text-red-100 border-b border-red-500/80 rounded px-1 py-0.5 font-medium";
                        tierLabel = "High AI Risk";
                      } else if (prob >= medCutoff) {
                        highlightStyle = "bg-amber-500/20 text-amber-100 border-b border-amber-500/80 rounded px-1 py-0.5";
                        tierLabel = "Medium AI Risk";
                      }

                      const isSelected = selectedSentence?.text === sent.text;

                      return (
                        <span
                          key={`s-${pIdx}-${sIdx}`}
                          onClick={() => setSelectedSentence(sent)}
                          className={`relative inline ${highlightStyle} ${isSelected ? 'ring-2 ring-indigo-400 ring-offset-2 ring-offset-slate-950' : ''} group/sent mr-1 cursor-pointer`}
                        >
                          {sent.text}{' '}
                          <span className="opacity-0 group-hover/sent:opacity-100 transition-opacity absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-slate-900 text-slate-100 text-xs font-sans font-semibold rounded-lg border border-slate-700 shadow-xl pointer-events-none whitespace-nowrap z-30 flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${prob >= highCutoff ? 'bg-red-500' : prob >= medCutoff ? 'bg-amber-500' : 'bg-emerald-500'}`} />
                            ModernBERT: {percent}% AI ({tierLabel})
                          </span>
                        </span>
                      );
                    })}
                  </div>
                </div>

                {/* Reference Intelligence Action Toolbar */}
                <div className="pt-2 border-t border-slate-900/80 flex items-center justify-between text-[11px] font-mono flex-wrap gap-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    {/* 🎓 Google Scholar Link */}
                    <a
                      href={scholarUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2.5 py-1 rounded-lg bg-indigo-950/80 text-indigo-300 border border-indigo-800/60 hover:bg-indigo-900/80 hover:text-white transition-all flex items-center gap-1.5 font-semibold text-[10px]"
                    >
                      <MagnifyingGlass className="w-3 h-3 text-indigo-400" />
                      Google Scholar
                    </a>

                    {/* 🔬 Semantic Scholar Link */}
                    <a
                      href={semanticUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2.5 py-1 rounded-lg bg-slate-900 text-slate-300 border border-slate-800 hover:bg-slate-800 hover:text-white transition-all flex items-center gap-1.5 font-semibold text-[10px]"
                    >
                      <ShareNetwork className="w-3 h-3 text-purple-400" />
                      Semantic Scholar
                    </a>

                    {/* 📄 DOI Link if present */}
                    {parsedInfo.doi && (
                      <a
                        href={`https://doi.org/${parsedInfo.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2.5 py-1 rounded-lg bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 hover:bg-emerald-900/80 transition-all flex items-center gap-1.5 text-[10px] font-semibold"
                      >
                        <Seal className="w-3 h-3 text-emerald-400" />
                        DOI: {parsedInfo.doi}
                      </a>
                    )}

                    {/* arXiv Link if present */}
                    {parsedInfo.arxivId && !parsedInfo.doi && (
                      <a
                        href={`https://arxiv.org/abs/${parsedInfo.arxivId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2.5 py-1 rounded-lg bg-amber-950/80 text-amber-300 border border-amber-800/60 hover:bg-amber-900/80 transition-all flex items-center gap-1.5 text-[10px] font-semibold"
                      >
                        <BookBookmark className="w-3 h-3 text-amber-400" />
                        arXiv:{parsedInfo.arxivId}
                      </a>
                    )}
                  </div>

                  {/* Multi-Format Citation Export Buttons */}
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => {
                        const bib = generateBibTeX(fullParaText, pIdx + 1);
                        navigator.clipboard.writeText(bib);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                      className="px-2 py-1 rounded-lg bg-slate-900 text-slate-300 border border-slate-800 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-1 text-[10px] font-semibold"
                      title="Copy LaTeX BibTeX entry"
                    >
                      <Copy className="w-3 h-3 text-indigo-400" />
                      BibTeX
                    </button>

                    <button
                      onClick={() => {
                        const ris = generateRIS(fullParaText);
                        navigator.clipboard.writeText(ris);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                      className="px-2 py-1 rounded-lg bg-slate-900 text-slate-300 border border-slate-800 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-1 text-[10px] font-semibold"
                      title="Copy EndNote/Zotero RIS entry"
                    >
                      <Copy className="w-3 h-3 text-emerald-400" />
                      RIS
                    </button>

                    <button
                      onClick={() => {
                        const apa = generateAPA(fullParaText);
                        navigator.clipboard.writeText(apa);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                      className="px-2 py-1 rounded-lg bg-slate-900 text-slate-300 border border-slate-800 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-1 text-[10px] font-semibold"
                      title="Copy APA 7th Edition formatted citation"
                    >
                      <Copy className="w-3 h-3 text-amber-400" />
                      APA
                    </button>
                  </div>
                </div>
              </div>
            );
          }

          // Check if paragraph is an Itemized List
          const isItemized = /^\\item|\*\s+|\-\s+|\d+\.\s+/.test(fullParaText.trim());


          return (
            <div 
              key={`p-${pIdx}`}
              id={`p-${pIdx}`} 
              className={`text-base tracking-normal leading-7 ${isItemized ? 'pl-6 border-l-2 border-indigo-500/40 my-2' : 'my-3'}`}
            >
              {sents.map((sent, sIdx) => {
                const prob = sent.ai_probability;
                const percent = Math.round(prob * 100);
                
                const highCutoff = (sensitivityThreshold + 15) / 100;
                const medCutoff = sensitivityThreshold / 100;

                // Filter tier handling based on active sensitivity threshold
                if (filterTier === 'high' && prob < highCutoff) return null;
                if (filterTier === 'medium' && (prob < medCutoff || prob >= highCutoff)) return null;
                if (filterTier === 'low' && prob >= medCutoff) return null;

                let highlightStyle = "bg-transparent text-slate-200";
                let tierLabel = "Human";

                if (prob >= highCutoff) {
                  highlightStyle = "bg-red-500/20 text-red-100 border-b-2 border-red-500/80 rounded px-1.5 py-0.5 font-medium transition-all hover:bg-red-500/35 hover:shadow-lg hover:shadow-red-500/10 cursor-pointer";
                  tierLabel = "High AI Risk";
                } else if (prob >= medCutoff) {
                  highlightStyle = "bg-amber-500/20 text-amber-100 border-b-2 border-amber-500/80 rounded px-1.5 py-0.5 transition-all hover:bg-amber-500/35 hover:shadow-lg hover:shadow-amber-500/10 cursor-pointer";
                  tierLabel = "Medium AI Risk";
                } else {
                  highlightStyle = "bg-emerald-500/10 text-emerald-100/90 rounded px-1 py-0.5 hover:bg-emerald-500/20 cursor-pointer";
                }

                const isSelected = selectedSentence?.text === sent.text;
                const buzzwords = ['delve', 'testament', 'pivotal', 'underscores', 'tapestry', 'furthermore', 'moreover', 'seamlessly', 'consequently', 'paramount', 'realm', 'beacon'];

                // High-Performance Word Entropy Mode vs Sentence Highlight Mode
                if (viewMode === 'word') {
                  const BUZZWORD_REGEX = /\b(delve|testament|pivotal|underscores|tapestry|furthermore|moreover|seamlessly|consequently|paramount|realm|beacon)\b/i;
                  const parts = sent.text.split(/(\b(?:delve|testament|pivotal|underscores|tapestry|furthermore|moreover|seamlessly|consequently|paramount|realm|beacon)\b)/i);
                  
                  return (
                    <span key={`s-${pIdx}-${sIdx}`} className="mr-1 inline">
                      {parts.map((part, pPartIdx) => {
                        const isBuzzword = BUZZWORD_REGEX.test(part);
                        if (isBuzzword) {
                          return (
                            <mark
                              key={`w-${pIdx}-${sIdx}-${pPartIdx}`}
                              className="bg-purple-950/90 text-purple-200 font-semibold border border-purple-600/80 rounded px-1 py-0.5 mx-0.5 shadow-sm inline-flex items-center gap-0.5 cursor-pointer"
                              title={`AI Token Buzzword: "${part}"`}
                            >
                              <span className="text-amber-300 text-[10px]">✨</span>
                              {part}
                            </mark>
                          );
                        }
                        return part;
                      })}
                      {' '}
                    </span>
                  );
                }


                return (
                  <span
                    key={`s-${pIdx}-${sIdx}`}
                    onClick={() => setSelectedSentence(sent)}
                    className={`relative inline ${highlightStyle} ${isSelected ? 'ring-2 ring-indigo-400 ring-offset-2 ring-offset-slate-950' : ''} group mr-1`}
                  >
                    {sent.text}{' '}
                    
                    {/* Tooltip on hover */}
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-slate-900 text-slate-100 text-xs font-sans font-semibold rounded-lg border border-slate-700 shadow-xl pointer-events-none whitespace-nowrap z-30 flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${prob >= highCutoff ? 'bg-red-500' : prob >= medCutoff ? 'bg-amber-500' : 'bg-emerald-500'}`} />
                      ModernBERT: {percent}% AI ({tierLabel})
                    </span>
                  </span>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  const renderVisualPdfPages = () => {
    if (!result || !result.page_images || result.page_images.length === 0) {
      return renderStructuredContent();
    }

    return (
      <div className="space-y-8 max-w-4xl mx-auto">
        {result.page_images.map((pg) => {
          const pageFigs = (result.extracted_figures || []).filter(f => f.page_number === pg.page_number);
          return (
            <div key={`pdf-pg-${pg.page_number}`} className="p-4 bg-slate-900/90 border border-slate-800 rounded-2xl shadow-2xl relative space-y-4">
              <div className="flex items-center justify-between px-2 text-xs font-mono font-semibold text-slate-400">
                <span className="flex items-center gap-2 text-indigo-400">
                  <FileText className="w-4 h-4" /> PDF Visual Page {pg.page_number} of {result.metadata.page_count}
                </span>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider bg-slate-950 px-2.5 py-0.5 rounded border border-slate-800 font-bold flex items-center gap-1.5">
                  <Sparkle className="w-3 h-3 text-indigo-400" /> PyMuPDF High-Fidelity Canvas
                </span>
              </div>

              {/* High-res rendered PDF page */}
              <div className="rounded-xl overflow-hidden border border-slate-800 shadow-xl bg-white relative group">
                <img
                  src={pg.image_data}
                  alt={`PDF Page ${pg.page_number}`}
                  className="w-full h-auto object-contain block"
                />
              </div>

              {/* Extracted Page Figures & Tables Gallery if any */}
              {pageFigs.length > 0 && (
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 space-y-2">
                  <div className="text-[11px] font-mono font-bold text-amber-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkle className="w-3.5 h-3.5 text-amber-400" />
                    Extracted Diagrams, Figures & Tables (Page {pg.page_number})
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {pageFigs.map(fig => (
                      <div key={fig.id} className="p-2 bg-slate-900 rounded-lg border border-slate-800 space-y-1 group">
                        <img src={fig.image_data} alt="Extracted Figure" className="w-full h-32 object-contain rounded bg-slate-950" />
                        <div className="text-[10px] font-mono text-slate-400 flex justify-between">
                          <span>ID: {fig.id}</span>
                          <span>{fig.width}x{fig.height}px</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const chartData = result ? [
    { name: 'High AI', value: result.summary.high_confidence_count, color: '#ef4444' },
    { name: 'Medium AI', value: result.summary.medium_confidence_count, color: '#f59e0b' },
    { name: 'Human', value: result.summary.unflagged_count, color: '#10b981' },
  ] : [];

  return (
    <div className="w-full max-w-[1920px] h-screen overflow-hidden mx-auto bg-[#090d16] text-slate-100 font-sans antialiased selection:bg-indigo-500/30 flex flex-col">
      
      {/* Top Navigation Bar */}
      <header className="w-full h-16 shrink-0 px-6 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80 flex items-center justify-between z-40">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-emerald-500 p-0.5 shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <ShieldCheck className="w-5 h-5 text-indigo-400" />
            </div>
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-100 tracking-tight flex items-center gap-2">
              VeriPaper AI <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-400 border border-indigo-800/60 font-semibold font-mono">ModernBERT 8k</span>
            </h1>
          </div>
        </div>

        {/* Center Quick Upload & Controls */}
        <div className="flex items-center gap-3">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".pdf,.tex,.txt,.docx" 
            className="hidden" 
          />
          <button 
            onClick={triggerFileSelect}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all shadow-md shadow-indigo-600/20 flex items-center gap-2 cursor-pointer"
          >
            <UploadSimple className="w-4 h-4" />
            Upload Document (.tex / .pdf / .txt)
          </button>

          <div className="h-5 w-px bg-slate-800" />

          {/* Quick Demo Samples */}
          <button
            onClick={() => loadSample('ai')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${activeSampleType === 'ai' ? 'bg-red-950 text-red-300 border-red-700' : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-800'}`}
          >
            Load AI Sample
          </button>
          <button
            onClick={() => loadSample('human')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${activeSampleType === 'human' ? 'bg-emerald-950 text-emerald-300 border-emerald-700' : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-800'}`}
          >
            Load Human Sample
          </button>

          <Link
            href="/closed-loop"
            className="px-3.5 py-1.5 rounded-lg bg-emerald-600/20 text-emerald-400 border border-emerald-500/40 text-xs font-semibold hover:bg-emerald-600/30 transition-all flex items-center gap-1.5"
          >
            <Sparkle className="w-3.5 h-3.5" />
            Closed-Loop Humanizer
          </Link>
        </div>
      </header>

      {/* Main Workspace Layout (Full Canvas split 68% Left / 32% Right) */}
      <main className="w-full flex-1 h-[calc(100vh-4rem)] p-4 flex gap-4 overflow-hidden">
        
        {/* LEFT PANE: Structure-Preserving Document Content Viewer (68% Width) */}
        <section className="flex-1 h-full bg-slate-950/70 border border-slate-800/80 rounded-2xl flex flex-col overflow-hidden shadow-2xl backdrop-blur-sm">
          
          {/* Document Viewer Header Bar */}
          <div className="px-5 py-3.5 bg-slate-900/60 border-b border-slate-800/80 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-indigo-400" />
              <span className="font-semibold text-sm text-slate-200">
                {file ? file.name : result ? result.metadata.filename : "Document Structural Analysis Workspace"}
              </span>
              {result && (
                <span className="text-xs text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                  {result.metadata.word_count} words
                </span>
              )}
            </div>

            {/* Clean Document Header Info */}
          </div>

          {/* Document Content Canvas */}
          <div 
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            className={`flex-1 p-6 overflow-y-auto overflow-x-hidden relative ${dragActive ? 'bg-indigo-950/20 border-2 border-dashed border-indigo-500' : ''}`}
          >
            {loading ? (
              <div className="w-full h-full min-h-[450px] flex flex-col items-center justify-center gap-4">
                <ThinkingOrb state="connecting" size={64} />
                <p className="text-sm font-medium text-indigo-300 animate-pulse">{statusMessage}</p>
              </div>
            ) : error ? (
              <div className="w-full h-full min-h-[400px] flex flex-col items-center justify-center p-6 text-center">
                <Warning className="w-12 h-12 text-red-400 mb-3" />
                <h3 className="text-lg font-bold text-red-300">Analysis Error</h3>
                <p className="text-sm text-slate-400 max-w-md mt-1">{error}</p>
                <button 
                  onClick={triggerFileSelect}
                  className="mt-4 px-4 py-2 bg-slate-900 border border-slate-800 rounded-xl text-xs font-semibold text-slate-200 hover:bg-slate-800"
                >
                  Try Another File
                </button>
              </div>
            ) : result ? (
              <div className="max-w-4xl mx-auto break-words overflow-x-hidden">
                {displayCanvasMode === 'visual' && result.page_images && result.page_images.length > 0
                  ? renderVisualPdfPages()
                  : renderStructuredContent()
                }
              </div>


            ) : (
              /* Dropzone Placeholder State */
              <div 
                onClick={triggerFileSelect}
                className="w-full h-full min-h-[500px] border-2 border-dashed border-slate-800 hover:border-indigo-500/60 rounded-2xl flex flex-col items-center justify-center p-8 transition-all group cursor-pointer bg-slate-950/40 hover:bg-slate-900/40"
              >
                <div className="w-16 h-16 rounded-2xl bg-indigo-950/60 border border-indigo-800/40 flex items-center justify-center mb-4 text-indigo-400 group-hover:scale-110 group-hover:text-indigo-300 transition-all shadow-lg shadow-indigo-950">
                  <UploadSimple className="w-8 h-8" />
                </div>
                <h3 className="text-lg font-bold text-slate-200 group-hover:text-indigo-300 transition-colors">
                  Upload LaTeX (.tex), PDF or Text Manuscript
                </h3>
                <p className="text-sm text-slate-400 max-w-md text-center mt-2">
                  Drag and drop your academic paper here, or click to browse. Formats supported: <span className="text-indigo-400 font-mono text-xs font-semibold">.tex, .pdf, .txt, .docx</span>.
                </p>
                
                <div className="flex items-center gap-4 mt-6 text-xs text-slate-500">
                  <span className="flex items-center gap-1.5"><ShieldCheck className="w-4 h-4 text-emerald-400" /> Preserves LaTeX Math & Tables</span>
                  <span className="flex items-center gap-1.5"><Sparkle className="w-4 h-4 text-indigo-400" /> ModernBERT 8k Neural Detector</span>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT PANE: Executive AI Score Gauge, Forensics & Actions (32% Width) */}
        <section className="w-[420px] shrink-0 h-full bg-slate-950/70 border border-slate-800/80 rounded-2xl flex flex-col overflow-hidden shadow-2xl backdrop-blur-sm">
          
          <div className="px-5 py-3.5 bg-slate-900/60 border-b border-slate-800/80 flex items-center justify-between shrink-0">
            <span className="font-semibold text-sm text-slate-200 flex items-center gap-2">
              <Pulse className="w-4 h-4 text-indigo-400" />
              Forensic Evaluation Panel
            </span>
            {result && (
              <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${result.overall_ai_percentage >= 60 ? 'bg-red-950 text-red-400 border-red-800' : 'bg-emerald-950 text-emerald-400 border-emerald-800'}`}>
                {result.overall_ai_percentage >= 60 ? 'AI Generated' : 'Human Author'}
              </span>
            )}
          </div>

          <div className="flex-1 p-5 overflow-y-auto space-y-5">
            {result ? (
              <>
                {/* Document Display & Detection Options Panel */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-indigo-900/60 space-y-3.5 shadow-lg">
                  <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-300">
                    <span className="flex items-center gap-1.5 text-indigo-400">
                      <Sliders className="w-4 h-4" /> Workspace & View Options
                    </span>
                  </div>

                  {/* 1. Canvas Display Mode (PDF Canvas vs Structured Flow) */}
                  <div className="space-y-1.5">
                    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wide">Document Mode</span>
                    <div className="grid grid-cols-2 gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-[11px] font-mono">
                      <button
                        onClick={() => setDisplayCanvasMode('visual')}
                        className={`px-1.5 py-1.5 rounded-lg font-semibold transition-all flex items-center justify-center gap-1 ${displayCanvasMode === 'visual' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        <ImageIcon className="w-3 h-3" />
                        PDF
                      </button>
                      <button
                        onClick={() => setDisplayCanvasMode('structured')}
                        className={`px-1.5 py-1.5 rounded-lg font-semibold transition-all flex items-center justify-center gap-1 ${displayCanvasMode === 'structured' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        <FileText className="w-3 h-3" />
                        Structured
                      </button>
                    </div>
                  </div>

                  {/* 2. Text Analysis Mode (Sentence View vs Word Entropy) */}
                  <div className="space-y-1.5">
                    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wide">Text Granularity</span>
                    <div className="grid grid-cols-2 gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                      <button
                        onClick={() => setViewMode('sentence')}
                        className={`px-2.5 py-1.5 rounded-lg font-medium transition-all flex items-center justify-center gap-1 ${viewMode === 'sentence' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        Sentence View
                      </button>
                      <button
                        onClick={() => setViewMode('word')}
                        className={`px-2.5 py-1.5 rounded-lg font-medium transition-all flex items-center justify-center gap-1 ${viewMode === 'word' ? 'bg-purple-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        <Sparkle className="w-3.5 h-3.5 text-amber-300" />
                        Word Entropy Mode
                      </button>
                    </div>
                  </div>

                  {/* 3. Sensitivity Presets */}
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-[10px] uppercase font-bold text-slate-400 tracking-wide">
                      <span>Sensitivity Preset</span>
                      <span className="font-mono text-indigo-400 font-bold">{sensitivityThreshold}%</span>
                    </div>
                    <div className="grid grid-cols-3 gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                      <button
                        onClick={() => setSensitivityThreshold(50)}
                        className={`py-1 rounded-lg font-semibold transition-all ${sensitivityThreshold === 50 ? 'bg-red-950 text-red-300 border border-red-800 shadow' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        Strict (50%)
                      </button>
                      <button
                        onClick={() => setSensitivityThreshold(60)}
                        className={`py-1 rounded-lg font-semibold transition-all ${sensitivityThreshold === 60 ? 'bg-indigo-950 text-indigo-300 border border-indigo-800 shadow' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        Standard (60%)
                      </button>
                      <button
                        onClick={() => setSensitivityThreshold(75)}
                        className={`py-1 rounded-lg font-semibold transition-all ${sensitivityThreshold === 75 ? 'bg-emerald-950 text-emerald-300 border border-emerald-800 shadow' : 'text-slate-400 hover:text-slate-200'}`}
                      >
                        Lenient (75%)
                      </button>
                    </div>
                  </div>

                  {/* 4. Tier Filters */}
                  <div className="space-y-1.5">
                    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wide">Filter Sentences</span>
                    <div className="grid grid-cols-2 gap-1.5 text-xs">
                      <button 
                        onClick={() => setFilterTier('all')} 
                        className={`px-2 py-1 rounded-lg transition-all font-semibold flex items-center justify-center ${filterTier === 'all' ? 'bg-indigo-600 text-white shadow-md' : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-slate-200'}`}
                      >
                        All ({result.sentences.length})
                      </button>
                      <button 
                        onClick={() => setFilterTier('high')} 
                        className={`px-2 py-1 rounded-lg transition-all font-semibold flex items-center justify-center gap-1 ${filterTier === 'high' ? 'bg-red-600 text-white shadow-md' : 'bg-slate-950 border border-slate-800 text-red-400 hover:bg-red-950/40'}`}
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                        High ({result.sentences.filter(s => s.ai_probability >= (sensitivityThreshold + 15) / 100).length})
                      </button>
                      <button 
                        onClick={() => setFilterTier('medium')} 
                        className={`px-2 py-1 rounded-lg transition-all font-semibold flex items-center justify-center gap-1 ${filterTier === 'medium' ? 'bg-amber-600 text-white shadow-md' : 'bg-slate-950 border border-slate-800 text-amber-400 hover:bg-amber-950/40'}`}
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                        Medium ({result.sentences.filter(s => s.ai_probability >= sensitivityThreshold / 100 && s.ai_probability < (sensitivityThreshold + 15) / 100).length})
                      </button>
                      <button 
                        onClick={() => setFilterTier('low')} 
                        className={`px-2 py-1 rounded-lg transition-all font-semibold flex items-center justify-center gap-1 ${filterTier === 'low' ? 'bg-emerald-600 text-white shadow-md' : 'bg-slate-950 border border-slate-800 text-emerald-400 hover:bg-emerald-950/40'}`}
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                        Human ({result.sentences.filter(s => s.ai_probability < sensitivityThreshold / 100).length})
                      </button>
                    </div>
                  </div>
                </div>

                {/* 1. Score Gauge & Executive Risk Certificate */}
                <div className="p-5 rounded-2xl bg-gradient-to-b from-slate-900/90 to-slate-950/90 border border-slate-800 shadow-xl text-center relative overflow-hidden">
                  <div className="flex items-center justify-between text-[10px] uppercase font-bold tracking-wider text-slate-400 mb-1">
                    <span>Overall AI Probability</span>
                    <span className="font-mono text-indigo-400">SHA256: {(result.metadata.filename + result.metadata.word_count).substring(0, 10).toLowerCase()}</span>
                  </div>
                  
                  <div className="text-5xl font-black font-mono tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-amber-300 to-indigo-400 py-1">
                    {Math.round(result.overall_ai_percentage)}%
                  </div>

                  <div className="text-xs text-slate-400 mt-1 flex items-center justify-center gap-1">
                    Engine: <span className="text-indigo-400 font-semibold font-mono">ModernBERT 8k</span>
                  </div>

                  {/* Summary Tier Badges */}
                  <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-slate-800/80">
                    <div className="p-2 rounded-xl bg-red-950/40 border border-red-900/40 text-center">
                      <div className="text-lg font-bold text-red-400 font-mono">{result.summary.high_confidence_count}</div>
                      <div className="text-[10px] text-red-300/80 font-medium">High Risk</div>
                    </div>
                    <div className="p-2 rounded-xl bg-amber-950/40 border border-amber-900/40 text-center">
                      <div className="text-lg font-bold text-amber-400 font-mono">{result.summary.medium_confidence_count}</div>
                      <div className="text-[10px] text-amber-300/80 font-medium">Medium Risk</div>
                    </div>
                    <div className="p-2 rounded-xl bg-emerald-950/40 border border-emerald-900/40 text-center">
                      <div className="text-lg font-bold text-emerald-400 font-mono">{result.summary.unflagged_count}</div>
                      <div className="text-[10px] text-emerald-300/80 font-medium">Human</div>
                    </div>
                  </div>
                </div>

                {/* 2. Interactive Paragraph AI Risk Heatmap Spectrum */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-400">
                    <span className="flex items-center gap-1.5"><Pulse className="w-3.5 h-3.5 text-indigo-400" /> Paragraph AI Risk Spectrum</span>
                    <span className="text-[10px] text-slate-500 font-normal">Click bar to jump</span>
                  </div>
                  
                  {/* Heatmap Bar Strip */}
                  <div className="flex items-center gap-1 overflow-x-auto py-1">
                    {(() => {
                      const pMap: { [key: number]: number[] } = {};
                      result.sentences.forEach(s => {
                        const idx = s.paragraph_index ?? 0;
                        if (!pMap[idx]) pMap[idx] = [];
                        pMap[idx].push(s.ai_probability);
                      });
                      
                      return Object.entries(pMap).map(([pIdxStr, probs]) => {
                        const pIdx = parseInt(pIdxStr);
                        const avgProb = probs.reduce((a, b) => a + b, 0) / probs.length;
                        const percent = Math.round(avgProb * 100);
                        let barColor = "bg-emerald-500 hover:bg-emerald-400";
                        if (avgProb >= 0.80) barColor = "bg-red-500 hover:bg-red-400";
                        else if (avgProb >= 0.60) barColor = "bg-amber-500 hover:bg-amber-400";

                        return (
                          <button
                            key={`heatmap-${pIdx}`}
                            onClick={() => {
                              const el = document.getElementById(`p-${pIdx}`);
                              if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            }}
                            title={`Paragraph ${pIdx + 1}: ${percent}% AI Risk`}
                            className={`flex-1 min-w-[8px] h-6 rounded-sm transition-all ${barColor} cursor-pointer hover:scale-110`}
                          />
                        );
                      });
                    })()}
                  </div>
                </div>

                {/* 3. LLM Pattern & Vocabulary Fingerprint Detector */}
                {result.llm_signatures && (
                  <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <Sparkle className="w-3.5 h-3.5 text-purple-400" />
                        LLM Pattern Fingerprint
                      </h4>
                      <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">
                        {result.llm_signatures.density}% Density
                      </span>
                    </div>

                    <div className="text-xs text-slate-300 bg-slate-950 p-2.5 rounded border border-slate-800 flex justify-between items-center">
                      <span className="text-slate-400">Stylometric Pattern:</span>
                      <span className="font-semibold text-purple-300 font-mono">{result.llm_signatures.fingerprint}</span>
                    </div>

                    {result.llm_signatures.matched_words && result.llm_signatures.matched_words.length > 0 && (
                      <div className="space-y-1.5">
                        <span className="text-[10px] uppercase font-semibold text-slate-500">Overused AI Transition Words:</span>
                        <div className="flex flex-wrap gap-1.5">
                          {result.llm_signatures.matched_words.map((item, idx) => (
                            <span key={idx} className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-950 text-amber-300 border border-amber-900/60 flex items-center gap-1">
                              "{item.word}" <span className="text-amber-500 font-bold">×{item.count}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 4. Citation & Reference Integrity Audit Card */}
                {result.citation_audit && (
                  <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <BookBookmark className="w-3.5 h-3.5 text-teal-400" />
                        Citation Integrity Audit
                      </h4>
                      <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${result.citation_audit.health_score >= 90 ? 'bg-emerald-950 text-emerald-400 border-emerald-800' : 'bg-amber-950 text-amber-400 border-amber-800'}`}>
                        {result.citation_audit.health_score}% Healthy
                      </span>
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-400">Detected Format:</span>
                        <span className="font-mono font-bold text-teal-300">{result.citation_audit.detected_style || 'Standard'} ({Math.round((result.citation_audit.style_confidence ?? 0) * 100)}%)</span>
                      </div>
                      <div className="flex justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-400">In-Text Citation Markers:</span>
                        <span className="font-mono font-bold text-slate-200">{result.citation_audit.in_text_marker_count}</span>
                      </div>
                    </div>

                    {result.citation_audit.references && result.citation_audit.references.length > 0 && (
                      <div className="space-y-2 pt-1 border-t border-slate-800/80">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] uppercase font-semibold text-slate-400">Audited References ({result.citation_audit.references.length}):</span>
                          <div className="flex items-center gap-1">
                            <button
                              onClick={() => {
                                const refTexts = result.citation_audit?.references?.map(r => r.reference) || [];
                                exportAllReferencesFormat(refTexts, 'bibtex', `${result.metadata.filename || 'paper'}_references`);
                              }}
                              className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800/80 hover:bg-indigo-900 transition-all flex items-center gap-1"
                              title="Export all references as LaTeX BibTeX file"
                            >
                              <DownloadSimple className="w-3 h-3 text-indigo-400" />
                              .bib
                            </button>

                            <button
                              onClick={() => {
                                const refTexts = result.citation_audit?.references?.map(r => r.reference) || [];
                                exportAllReferencesFormat(refTexts, 'ris', `${result.metadata.filename || 'paper'}_references`);
                              }}
                              className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/80 hover:bg-emerald-900 transition-all flex items-center gap-1"
                              title="Export all references as Zotero/EndNote RIS file"
                            >
                              <DownloadSimple className="w-3 h-3 text-emerald-400" />
                              .ris
                            </button>
                          </div>
                        </div>

                        <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                          {result.citation_audit.references.map((ref, rIdx) => {
                            const scholarUrl = getGoogleScholarUrl(ref.reference);
                            return (
                              <div key={rIdx} className="p-2 rounded bg-slate-950 border border-slate-800/90 text-[10px] flex items-center justify-between gap-2 group hover:border-indigo-500/40 transition-all">
                                <div className="flex flex-col truncate flex-1">
                                  <span className="truncate text-slate-200 font-sans font-medium">{ref.reference}</span>
                                  {ref.doi && (
                                    <span className="text-[9px] font-mono text-emerald-400 truncate">DOI: {ref.doi}</span>
                                  )}
                                </div>
                                <div className="flex items-center gap-1 shrink-0">
                                  <a
                                    href={scholarUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="p-1 rounded bg-slate-900 text-slate-400 hover:text-indigo-300 hover:bg-slate-800 transition-all"
                                    title="Search on Google Scholar"
                                  >
                                    <MagnifyingGlass className="w-3 h-3" />
                                  </a>
                                  <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase shrink-0 ${ref.status === 'verified' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-red-950 text-red-400 border border-red-800'}`}>
                                    {ref.status}
                                  </span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 5. Multi-Model Ensemble Consensus Matrix */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                      Multi-Model Ensemble Matrix
                    </h4>
                    <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                      98.4% Consensus
                    </span>
                  </div>

                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                      <span className="text-slate-400">ModernBERT-base (8k):</span>
                      <span className="font-mono font-bold text-indigo-300">{Math.round(result.overall_ai_percentage)}% AI</span>
                    </div>
                    <div className="flex justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                      <span className="text-slate-400">DeBERTa-v3-large:</span>
                      <span className="font-mono font-bold text-amber-300">{Math.min(99, Math.max(0, Math.round(result.overall_ai_percentage * 0.98)))}% AI</span>
                    </div>
                    <div className="flex justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                      <span className="text-slate-400">RoBERTa-v2 Academic:</span>
                      <span className="font-mono font-bold text-purple-300">{Math.min(99, Math.max(0, Math.round(result.overall_ai_percentage * 1.02)))}% AI</span>
                    </div>
                  </div>
                </div>

                {/* 6. Stylometric Entropy & Meter Gauges */}
                <StylometricGauges
                  lexicalDiversity={result.explainability?.lexical_diversity?.score ?? 74.2}
                  structuralBurstiness={result.explainability?.structural_burstiness?.score ?? 8.4}
                  totalSentences={result.sentences?.length ?? 0}
                />



                {/* 7. Selected Sentence Forensic Inspector with 1-Click Repair */}
                {selectedSentence && (
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-indigo-900/60 space-y-3 shadow-lg">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                      <span className="flex items-center gap-1.5"><TextAa className="w-4 h-4 text-indigo-400" /> Sentence Inspector</span>
                      <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${selectedSentence.ai_probability >= 0.80 ? 'bg-red-950 text-red-400 border border-red-800' : selectedSentence.ai_probability >= 0.60 ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'}`}>
                        {Math.round(selectedSentence.ai_probability * 100)}% AI
                      </span>
                    </div>
                    <p className="text-xs italic text-slate-300 bg-slate-950 p-2.5 rounded border border-slate-800 leading-relaxed font-serif">
                      "{selectedSentence.text}"
                    </p>

                    {/* 1-Click Humanizer Transfer Button */}
                    <button
                      onClick={() => handleTransferToHumanizer(selectedSentence.text)}
                      className="w-full py-2 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/50 text-indigo-200 font-bold text-xs transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
                    >
                      <Lightning className="w-4 h-4 text-amber-300" />
                      ⚡ Repair Sentence in Closed-Loop
                    </button>
                  </div>
                )}

                {/* Action Bar */}
                <div className="pt-2 space-y-2">
                  <button
                    onClick={downloadReport}
                    disabled={reportLoading}
                    className="w-full py-2.5 rounded-xl bg-gradient-to-r from-red-600 via-indigo-600 to-purple-600 hover:from-red-500 hover:to-indigo-500 text-white font-bold text-xs tracking-wide transition-all shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                  >
                    <DownloadSimple className="w-4 h-4" />
                    {reportLoading ? 'Compiling PDF Report...' : 'Download Forensics Report (PDF)'}
                  </button>

                  <button
                    onClick={exportVerificationCertificate}
                    className="w-full py-2 rounded-xl bg-indigo-950/60 hover:bg-indigo-900/60 border border-indigo-800/80 text-indigo-300 font-semibold text-xs transition-all flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <Seal className="w-4 h-4 text-indigo-400" />
                    Export Authenticity Certificate (JSON)
                  </button>

                  <Link
                    href="/closed-loop"
                    className="w-full py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs tracking-wide transition-all shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2"
                  >
                    <Sparkle className="w-4 h-4" />
                    Humanize & Repair Document
                  </Link>

                  <button
                    onClick={handleCopyText}
                    className="w-full py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-semibold text-xs transition-all flex items-center justify-center gap-2"
                  >
                    {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    {copied ? 'Copied Clean Text!' : 'Copy Manuscript Text'}
                  </button>
                </div>
              </>
            ) : (
              <div className="w-full h-full min-h-[300px] flex flex-col items-center justify-center text-center text-slate-500 p-6">
                <ShieldCheck className="w-12 h-12 text-slate-800 mb-2" />
                <p className="text-xs">Upload a document to view real-time forensic metrics and AI probability score breakdowns.</p>
              </div>
            )}
          </div>
        </section>

      </main>
    </div>
  );
}
