'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  IdentificationCard,
  Quotes
} from '@phosphor-icons/react';
import { ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

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

interface AnalysisResult {
  metadata: {
    filename: string;
    page_count: number;
    word_count: number;
  };
  overall_ai_percentage: number;
  sentences: Sentence[];
  explainability: {
    lexical_diversity: { score: number; label: string };
    structural_burstiness: { score: number; label: string };
  };
  llm_signatures: {
    density: number;
    fingerprint: string;
    matched_words: Array<{ word: string; count: number; model: string }>;
  };
  citation_audit: {
    health_score: number;
    detected_style?: string;
    style_confidence?: number;
    in_text_marker_count?: number;
    references: Reference[];
    unmatched_citations?: Array<{ marker_text: string; referenced_value: string }>;
    unreferenced_entries?: Array<{ entry_number_or_index: string; raw_text: string }>;
  };
  summary: {
    high_confidence_count: number;
    medium_confidence_count: number;
    unflagged_count: number;
  };
  is_simulated?: boolean;
}

export default function DocumentVerificationInstrument() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedSentence, setSelectedSentence] = useState<Sentence | null>(null);
  const [expandedCitationIdx, setExpandedCitationIdx] = useState<number | null>(null);
  const [hoveredCitationKey, setHoveredCitationKey] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("Initializing analysis...");
  const [shouldAnimate, setShouldAnimate] = useState<boolean>(true);
  const [reportLoading, setReportLoading] = useState<boolean>(false);
  const [activeSampleType, setActiveSampleType] = useState<'human' | 'ai' | null>(null);

  // Helper to extract citation keys (e.g. "[4]") from bibliography lines
  const getCitationKey = (reference: string): string | null => {
    const match = reference.trim().match(/^\[(\d+)\]/);
    return match ? `[${match[1]}]` : null;
  };

  // Helper to extract citation title from bibliography line for targeted Scholar search
  const getScholarQuery = (reference: string): string => {
    const match = reference.match(/["““]([^""“”]+)["””]/);
    if (match && match[1].trim()) {
      const cleanTitle = match[1].trim().replace(/[ ,.!?]+$/, "");
      return `"${cleanTitle}"`;
    }
    return reference;
  };
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check prefers-reduced-motion
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setShouldAnimate(!mediaQuery.matches);
  }, []);

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
      const ext = droppedFile.name.split('.').pop()?.toLowerCase();
      const validExtensions = ['pdf', 'docx', 'doc'];
      const validMimeTypes = [
        'application/pdf', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword'
      ];
      
      if ((ext && validExtensions.includes(ext)) || validMimeTypes.includes(droppedFile.type)) {
        setFile(droppedFile);
        setActiveSampleType(null);
        await analyzeDocument(droppedFile);
      } else {
        setError("Unsupported file format. Only PDF, DOCX, and DOC documents are supported.");
      }
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

  // Call backend analysis API
  const analyzeDocument = async (targetFile: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedSentence(null);
    
    const steps = [
      "Extracting document text layers...",
      "Segmenting syntax structure and sentences...",
      "Evaluating stylistic entropy and lexical diversity...",
      "Running RoBERTa neural classification on sentences...",
      "Auditing bibliography against CrossRef API index...",
      "Finalizing document forensics report..."
    ];

    let currentStep = 0;
    setStatusMessage(steps[0]);
    
    const stepInterval = setInterval(() => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        setStatusMessage(steps[currentStep]);
      }
    }, 1800);

    try {
      const formData = new FormData();
      formData.append("file", targetFile);

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Verification failed. Please check the backend connection.");
      }

      const data = await response.json();
      setResult(data);
      
      // Auto-select first sentence if available
      if (data.sentences && data.sentences.length > 0) {
        setSelectedSentence(data.sentences[0]);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unexpected error occurred during manuscript analysis.");
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  // Call backend sample analysis API
  const loadSample = async (type: 'human' | 'ai') => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedSentence(null);
    setActiveSampleType(type);
    
    const steps = [
      "Locating local sample manuscript...",
      "Extracting text layers from PDF pages...",
      "Running RoBERTa neural classification on sentences...",
      "Auditing bibliography against CrossRef API index...",
      "Finalizing document forensics report..."
    ];

    let currentStep = 0;
    setStatusMessage(steps[0]);
    
    const stepInterval = setInterval(() => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        setStatusMessage(steps[currentStep]);
      }
    }, 1800);

    try {
      const response = await fetch(`http://127.0.0.1:8000/sample?type=${type}`, {
        method: "GET",
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Sample analysis failed. Please check the backend connection.");
      }

      const data = await response.json();
      setResult(data);
      
      // Auto-select first sentence if available
      if (data.sentences && data.sentences.length > 0) {
        setSelectedSentence(data.sentences[0]);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unexpected error occurred loading sample paper.");
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    setReportLoading(true);
    try {
      const formData = new FormData();
      let url = "http://127.0.0.1:8000/report";
      
      if (activeSampleType) {
        url += `?sample_type=${activeSampleType}`;
      } else if (file) {
        formData.append("file", file);
      } else {
        throw new Error("No active document or sample found to generate report.");
      }

      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errMsg = "Failed to compile the PDF forensics report.";
        try {
          const errData = await response.json();
          errMsg = errData.detail || errMsg;
        } catch (e) {}
        throw new Error(errMsg);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      
      let filename = "VeriPaper_Forensics_Report.pdf";
      if (activeSampleType) {
        filename = activeSampleType === "human" ? "EJ1172284_VeriPaper_Report.pdf" : "LLM_Survey_VeriPaper_Report.pdf";
      } else if (file) {
        const namePart = file.name.split('.').slice(0, -1).join('_');
        filename = `${namePart}_VeriPaper_Report.pdf`;
      }
      
      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
      
    } catch (err: any) {
      console.error(err);
      alert(err.message || "An unexpected error occurred while downloading the PDF report.");
    } finally {
      setReportLoading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setSelectedSentence(null);
    setActiveSampleType(null);
  };

  // Group sentences by paragraph index
  const getParagraphs = (): Sentence[][] => {
    if (!result) return [];
    
    const paragraphs: { [key: number]: Sentence[] } = {};
    result.sentences.forEach(sent => {
      const pIdx = sent.paragraph_index || 0;
      if (!paragraphs[pIdx]) {
        paragraphs[pIdx] = [];
      }
      paragraphs[pIdx].push(sent);
    });
    
    return Object.keys(paragraphs)
      .sort((a, b) => Number(a) - Number(b))
      .map(key => paragraphs[Number(key)] || []);
  };

  // Mathematical generation of organic fluted wax seal path
  const getRosettePath = () => {
    let d = "";
    const points = 32;
    for (let i = 0; i < points; i++) {
      const angle = (i * 2 * Math.PI) / points;
      const r = i % 2 === 0 ? 48 : 44; // oscillates radii
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

  // Render explainability meter styles for Recharts
  const renderGauge = (score: number, maxScore: number, type: 'lexical' | 'burstiness') => {
    let color = '#4B6A57'; // verified verdigris
    if (type === 'lexical') {
      if (score < 58.0) color = '#B23A2E'; // flag-high red
      else if (score < 68.0) color = '#B8862E'; // flag-medium amber
    } else {
      if (score < 30.0) color = '#B23A2E';
      else if (score < 45.0) color = '#B8862E';
    }
    
    const data = [
      { name: 'score', value: Math.min(score, maxScore), fill: color },
      { name: 'remainder', value: Math.max(0, maxScore - score), fill: '#DCD4C0' }
    ];
    
    return (
      <div className="relative w-full h-24 flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
            <Pie
              data={data}
              startAngle={180}
              endAngle={0}
              innerRadius="75%"
              outerRadius="95%"
              dataKey="value"
              stroke="none"
              cx="50%"
              cy="100%"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute bottom-0 flex flex-col items-center">
          <span className="font-mono text-lg font-bold text-[#21242B] leading-none">
            {score.toFixed(1)}{type === 'lexical' ? '%' : ''}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#FAF8F2] text-[#21242B] font-body selection:bg-[#7A2331]/10 selection:text-[#7A2331] antialiased">
      
      {/* Header bar */}
      <header className="flex items-center justify-between px-10 py-5 border-b border-[#DCD4C0] bg-[#FAF8F2] shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-display font-medium tracking-wide text-[#7A2331] flex items-center gap-2 select-none">
            {/* Custom mini rosette seal */}
            <svg className="w-6 h-6 animate-pulse" viewBox="0 0 100 100" fill="currentColor">
              <path d={getRosettePath()} />
            </svg>
            VeriPaper AI
          </span>
          <span className="text-[10px] font-mono tracking-widest text-[#7A2331] bg-transparent border border-[#DCD4C0] px-2 py-0.5 rounded uppercase">
            ROBERTA-V2
          </span>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
            {loading ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-[#B8862E] animate-pulse"></span>
                <span>AUDITING MANUSCRIPT...</span>
              </>
            ) : result ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-[#4B6A57]"></span>
                <span className="text-[#4B6A57] font-semibold">VERIFICATION DOSSIER READY</span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                <span>INSTRUMENT STANDBY</span>
              </>
            )}
          </div>
          
          {(result || error) && (
            <button 
              onClick={resetUpload}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-mono font-medium rounded border border-[#DCD4C0] hover:border-[#7A2331] hover:text-[#7A2331] transition-all bg-transparent cursor-pointer"
            >
              <ArrowCounterClockwise weight="duotone" className="w-3.5 h-3.5" />
              NEW AUDIT
            </button>
          )}
        </div>
      </header>

      {/* Main workspace */}
      <main className="flex-1 flex flex-col md:flex-row p-10 gap-10 overflow-hidden max-w-[1600px] w-full mx-auto">
        
        {/* Loading state overlay */}
        {loading && (
          <div className="flex-1 flex flex-col items-center justify-center bg-[#FAF8F2] border border-[#DCD4C0] shadow-paper-shadow rounded-lg p-16">
            <div className="relative w-16 h-16 mb-8 flex items-center justify-center">
              <svg className="w-full h-full text-[#7A2331] animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </div>
            <p className="text-xl font-display font-medium text-[#7A2331] mb-2">{statusMessage}</p>
            <p className="text-xs font-mono text-slate-400">ANALYZING MANUSCRIPT INK AND BIBLIOGRAPHY</p>
          </div>
        )}

        {/* Empty Upload State */}
        {!loading && !result && (
          <div className="flex-1 flex flex-col md:flex-row gap-10">
            
            {/* Upload Zone */}
            <div 
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={triggerFileSelect}
              className={`flex-1 flex flex-col items-center justify-center border border-dashed rounded-lg p-16 text-center transition-all cursor-pointer bg-white shadow-paper-shadow
                ${dragActive ? 'border-[#7A2331] bg-[#FAF8F2]' : 'border-[#DCD4C0] hover:border-[#7A2331] hover:bg-slate-50/20'}`}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept=".pdf,.docx,.doc"
                className="hidden" 
                onChange={handleFileChange}
              />
              
              <div className="w-20 h-20 rounded-full bg-[#7A2331]/5 flex items-center justify-center mb-8">
                <UploadSimple weight="duotone" className="w-10 h-10 text-[#7A2331]" />
              </div>
              
              <h3 className="text-2xl font-display font-medium text-slate-900 mb-2">Submit Academic Manuscript for Forensics</h3>
              <p className="text-sm text-slate-500 mb-8 max-w-md leading-relaxed">
                Drag and drop your academic PDF, DOCX, or DOC. Our neural models will perform stratified sentence audit, lexical style tracing, and citation validation.
              </p>
              
              <div className="text-xs font-mono text-slate-500 bg-[#FAF8F2] px-4 py-2 border border-[#DCD4C0] rounded">
                PDF, DOCX, OR DOC FORMAT REQUIRED • LIMIT 25MB
              </div>

              <div className="mt-8 flex gap-3 flex-wrap justify-center z-10 animate-fade-in" onClick={(e) => e.stopPropagation()}>
                <button
                  onClick={() => loadSample('human')}
                  className="px-4 py-2.5 text-xs font-mono font-medium border border-[#DCD4C0] rounded hover:border-[#7A2331] hover:text-[#7A2331] transition-all bg-white hover:bg-slate-50/50 shadow-sm cursor-pointer uppercase"
                >
                  Load Sample Human Paper
                </button>
                <button
                  onClick={() => loadSample('ai')}
                  className="px-4 py-2.5 text-xs font-mono font-medium border border-[#DCD4C0] rounded hover:border-[#7A2331] hover:text-[#7A2331] transition-all bg-white hover:bg-slate-50/50 shadow-sm cursor-pointer uppercase"
                >
                  Load Sample AI Paper
                </button>
              </div>

              {error && (
                <div className="mt-8 px-5 py-3 border border-red-200 bg-red-50/50 rounded text-xs text-red-700 font-mono font-medium">
                  {error}
                </div>
              )}
            </div>

            {/* Sidebar Placeholder */}
            <div className="w-full md:w-[400px] shrink-0 flex flex-col gap-8 opacity-40 select-none pointer-events-none">
              <div className="bg-white border border-[#DCD4C0] shadow-paper-shadow rounded-lg p-8">
                <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-4">Verification Certificate</h4>
                <div className="h-32 border border-dashed border-[#DCD4C0] rounded flex items-center justify-center text-xs text-slate-400 font-mono bg-[#FAF8F2]/50">
                  AWAITING UPLOAD
                </div>
              </div>
              <div className="bg-white border border-[#DCD4C0] shadow-paper-shadow rounded-lg p-8">
                <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-4">Inspection Report</h4>
                <div className="h-48 border border-dashed border-[#DCD4C0] rounded flex items-center justify-center text-xs text-slate-400 font-mono bg-[#FAF8F2]/50">
                  AWAITING UPLOAD
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Populated Report State */}
        {!loading && result && (
          <div className="flex-1 flex flex-col md:flex-row gap-10 overflow-hidden h-full">
            
            {/* Left Column: Interactive Document Reading Pane */}
            <div className="flex-1 flex flex-col bg-white border border-[#DCD4C0] shadow-paper-shadow rounded-lg p-10 max-h-[calc(100vh-140px)] overflow-y-auto relative">
              <div className="border-b border-[#DCD4C0] pb-5 mb-8 flex justify-between items-center shrink-0">
                <h2 className="text-xl font-display font-medium text-slate-900 flex items-center gap-2">
                  <FileText weight="duotone" className="w-6 h-6 text-[#7A2331]" />
                  Manuscript Reading Pane
                </h2>
                
                {result.is_simulated && (
                  <span className="text-[10px] font-mono font-semibold tracking-wider text-amber-800 bg-amber-50 border border-amber-200 px-3 py-1 rounded">
                    SIMULATION DEMO ACTIVE
                  </span>
                )}
              </div>

              {/* Scrollable text section with structured paragraphs */}
              <div className="flex-1 space-y-8 text-[#21242B] text-[17px] leading-relaxed pr-3 select-text max-w-4xl mx-auto pl-10 relative">
                {getParagraphs().map((paragraph, pIdx) => {
                  // Find highest confidence flag in this paragraph
                  let highestFlag: 'high' | 'medium' | 'none' = 'none';
                  for (const sent of paragraph) {
                    if (sent.confidence_tier === 'high') {
                      highestFlag = 'high';
                    } else if (sent.confidence_tier === 'medium' && highestFlag !== 'high') {
                      highestFlag = 'medium';
                    }
                  }

                  // Render left margin bracket glyph
                  let bracketColor = "text-slate-200";
                  if (highestFlag === 'high') bracketColor = "text-[#B23A2E]";
                  else if (highestFlag === 'medium') bracketColor = "text-[#B8862E]";

                  return (
                    <div key={pIdx} className="relative group/p">
                      
                      {/* Fade-in staggered left margin bracket */}
                      {highestFlag !== 'none' && (
                        <motion.span 
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ 
                            delay: shouldAnimate ? pIdx * 0.05 : 0, 
                            type: "spring", 
                            stiffness: 180, 
                            damping: 15 
                          }}
                          className={`absolute -left-8 top-1 font-mono text-base font-bold select-none ${bracketColor}`}
                          title={`Paragraph contains AI indicators`}
                        >
                          ⌐
                        </motion.span>
                      )}

                      <p className="indent-8 text-justify">
                        {paragraph.map((sent, sIdx) => {
                          const isSelected = selectedSentence?.text === sent.text;
                          
                          // Custom dotted underlines per confidence tier
                          let decorationClass = "";
                          if (sent.confidence_tier === 'high') {
                            decorationClass = isSelected 
                              ? "border-b-2 border-dashed border-[#B23A2E] bg-[#B23A2E]/5 text-slate-900" 
                              : "border-b border-dashed border-[#B23A2E]/80 hover:bg-[#B23A2E]/5 text-slate-900";
                          } else if (sent.confidence_tier === 'medium') {
                            decorationClass = isSelected 
                              ? "border-b-2 border-dashed border-[#B8862E] bg-[#B8862E]/5 text-slate-900" 
                              : "border-b border-dashed border-[#B8862E]/80 hover:bg-[#B8862E]/5 text-slate-900";
                          } else {
                            decorationClass = isSelected 
                              ? "bg-slate-100 ring-1 ring-[#DCD4C0] text-slate-800" 
                              : "hover:bg-slate-50/80 text-slate-700";
                          }

                          const isCitationHighlighted = hoveredCitationKey && sent.text.includes(hoveredCitationKey);
                          return (
                            <span 
                              key={sIdx}
                              onMouseEnter={() => setSelectedSentence(sent)}
                              onClick={() => setSelectedSentence(sent)}
                              className={`inline rounded transition-all duration-150 cursor-pointer mx-0.5 py-0.5 select-text 
                                ${isCitationHighlighted ? 'bg-[#7A2331]/10 ring-2 ring-[#7A2331]/30 font-semibold scale-[1.01]' : ''} 
                                ${decorationClass}`}
                            >
                              {sent.text}{" "}
                            </span>
                          );
                        })}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right Column: Sticky Inspection Dossier */}
            <div className="w-full md:w-[400px] shrink-0 flex flex-col overflow-y-auto max-h-[calc(100vh-140px)] pr-2 select-none">
              
              {/* Dossier Container */}
              <div className="bg-white border border-[#DCD4C0] shadow-paper-shadow rounded-lg flex flex-col p-8 divide-y divide-[#DCD4C0]">
                
                {/* 1. Verification Certificate Seal */}
                <div className="pb-8 flex flex-col items-center">
                  <span className="text-sm font-mono tracking-[0.05em] text-slate-400 uppercase mb-5">VERIFICATION CERTIFICATE</span>
                  
                  {/* Wax Seal Rosette stamp */}
                  <motion.div 
                    variants={sealVariants}
                    initial="initial"
                    animate="animate"
                    className="relative w-28 h-28 flex items-center justify-center cursor-pointer select-none"
                  >
                    {/* Shadow pulse stamp overlay */}
                    <motion.div 
                      variants={shadowVariants}
                      initial="initial"
                      animate="animate"
                      className="absolute inset-0 rounded-full"
                    />
                    
                    {/* Outer wavy wax path */}
                    <svg className="absolute w-full h-full text-[#7A2331] drop-shadow-[0_2px_4px_rgba(122,35,49,0.15)]" viewBox="0 0 100 100" fill="currentColor">
                      <path d={getRosettePath()} />
                      <circle cx="50" cy="50" r="37" fill="#FAF8F2" stroke="#7A2331" strokeWidth="1.5" strokeDasharray="3 3" />
                    </svg>
                    
                    {/* Center text score */}
                    <div className="absolute flex flex-col items-center justify-center z-10 leading-none">
                      <span className="font-display text-[26px] font-bold text-[#7A2331] leading-none">{result.overall_ai_percentage.toFixed(0)}%</span>
                      <span className="font-mono text-[10px] text-[#7A2331] tracking-wider font-semibold mt-0.5">AI TEXT</span>
                    </div>
                  </motion.div>

                  {/* Rosette certificate subtext */}
                  <div className="mt-5 text-center">
                    <span className="font-mono text-md text-[#7A2331] block font-bold tracking-widest uppercase mb-1">
                      {result.overall_ai_percentage >= 50 ? "PROBABLE SYNTHETIC SIGNATURE" : "CONFIRMED SCHOLARLY INK"}
                    </span>
                    <span className="font-mono text-sm text-slate-400 block truncate max-w-[320px]" title={result.metadata.filename}>
                      DOCID: {result.metadata.filename}
                    </span>
                    
                    {/* Download PDF report button */}
                    <button
                      disabled={reportLoading}
                      onClick={downloadReport}
                      className="mt-6 w-full py-2.5 px-4 rounded bg-[#7A2331] hover:bg-[#5D1924] text-[#FAF8F2] font-mono text-xs font-semibold tracking-wider transition-all disabled:bg-slate-300 disabled:text-slate-500 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-sm"
                    >
                      {reportLoading ? (
                        <>
                          <span className="animate-spin rounded-full h-3.5 w-3.5 border-2 border-[#FAF8F2] border-t-transparent"></span>
                           GENERATING REPORT...
                        </>
                      ) : (
                        "DOWNLOAD REPORT (PDF)"
                      )}
                    </button>
                  </div>
                </div>

                {/* 2. Metadata Readings */}
                <div className="py-6 space-y-[12px] font-mono text-slate-500">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">MANUSCRIPT VOLUME:</span>
                    <span className="text-base font-semibold text-slate-800">{result.metadata.page_count} PAGES</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">WORD COUNT SEGMENTS:</span>
                    <span className="text-base font-semibold text-slate-800">{result.metadata.word_count} WORDS</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">FORENSIC TIMESTAMP:</span>
                    <span className="text-base font-semibold text-slate-800 uppercase">JULY 2026</span>
                  </div>
                </div>

                {/* 3. Sentinel Sentence Detail Inspector */}
                <div className="py-6">
                  <div className="flex items-center gap-1.5 text-md font-display font-medium text-slate-900 mb-4 tracking-wide uppercase">
                    <Quotes weight="duotone" className="w-4 h-4 text-[#7A2331]" />
                    Sentinel Inspector
                  </div>
                  
                  {selectedSentence ? (
                    <div className="space-y-4">
                      <div className="text-base italic text-slate-800 bg-[#FAF8F2] p-5 border border-[#DCD4C0] rounded font-serif leading-[1.6] relative max-h-[120px] overflow-y-auto">
                        <span className="absolute left-1.5 top-1 text-slate-300 font-mono text-base font-bold">“</span>
                        <span className="pl-3 block">"{selectedSentence.text}"</span>
                      </div>
                      
                      <div>
                        <div className="flex items-center justify-between font-mono text-slate-500 mb-1.5">
                          <span className="text-sm">SYNTHESIS PROBABILITY:</span>
                          <span className={`text-base font-bold ${
                            selectedSentence.confidence_tier === 'high' ? 'text-[#B23A2E]' :
                            selectedSentence.confidence_tier === 'medium' ? 'text-[#B8862E]' : 'text-[#4B6A57]'
                          }`}>{(selectedSentence.ai_probability * 100).toFixed(1)}%</span>
                        </div>
                        {/* Custom horizontal rule trace bar */}
                        <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-500 ease-out
                              ${selectedSentence.confidence_tier === 'high' ? 'bg-[#B23A2E]' :
                                selectedSentence.confidence_tier === 'medium' ? 'bg-[#B8862E]' : 'bg-[#4B6A57]'}`}
                            style={{ width: `${selectedSentence.ai_probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs italic text-slate-400 py-3 text-center">
                      Select or hover over any underlined sentence in the viewer to trace its neural composition.
                    </div>
                  )}
                </div>

                {/* 4. Explainability Diagnostics */}
                <div className="py-6 space-y-5">
                  <div className="flex items-center gap-1.5 text-md font-display font-medium text-slate-900 tracking-wide uppercase">
                    <Pulse weight="duotone" className="w-4 h-4 text-[#7A2331]" />
                    Stylistic Entropy
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    {/* Gauge 1: Lexical Diversity */}
                    <div className="flex flex-col items-center p-3 bg-slate-50/50 border border-[#DCD4C0]/50 rounded">
                      <span className="font-mono text-sm text-slate-400 font-bold tracking-wider uppercase mb-1">Lexical Diversity</span>
                      {renderGauge(result.explainability.lexical_diversity.score, 100, 'lexical')}
                      <span className="font-mono text-sm text-slate-500 text-center block mt-1 leading-snug">
                        {result.explainability.lexical_diversity.label}
                      </span>
                    </div>
                    
                    {/* Gauge 2: Structural Burstiness */}
                    <div className="flex flex-col items-center p-3 bg-slate-50/50 border border-[#DCD4C0]/50 rounded">
                      <span className="font-mono text-sm text-slate-400 font-bold tracking-wider uppercase mb-1">Burstiness (CV)</span>
                      {renderGauge(result.explainability.structural_burstiness.score, 120, 'burstiness')}
                      <span className="font-mono text-sm text-slate-500 text-center block mt-1 leading-snug">
                        {result.explainability.structural_burstiness.label}
                      </span>
                    </div>
                  </div>
                </div>

                {/* 5. LLM Writing Fingerprint */}
                <div className="py-6">
                  <div className="flex items-center gap-1.5 text-md font-display font-medium text-slate-900 mb-4 tracking-wide uppercase">
                    <TextAa weight="duotone" className="w-4 h-4 text-[#7A2331]" />
                    Writing Fingerprint
                  </div>
                  
                  <div className="p-3 bg-[#FAF8F2] border border-[#DCD4C0] rounded">
                    <div className="flex justify-between items-center mb-1 font-mono text-slate-500">
                      <span className="text-sm">TRANSITION KEYWORDS:</span>
                      <span className={`text-sm font-mono font-bold ${result.llm_signatures.density > 4.5 ? 'text-[#B23A2E]' : result.llm_signatures.density > 1.5 ? 'text-[#B8862E]' : 'text-[#4B6A57]'}`}>
                        {result.llm_signatures.density.toFixed(1)} / 1K WORDS
                      </span>
                    </div>
                    <div className="text-base font-bold text-slate-800 mb-3">
                      {result.llm_signatures.fingerprint}
                    </div>
                    
                    {/* Model-specific tags */}
                    {result.llm_signatures.matched_words.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {result.llm_signatures.matched_words.map((w, idx) => (
                          <span 
                            key={idx} 
                            className={`text-sm font-mono px-[8px] py-[4px] rounded border 
                              ${w.model === 'Claude' ? 'bg-purple-50 text-purple-700 border-purple-200/60' : 'bg-sky-50 text-sky-700 border-sky-200/60'}`}
                            title={`Found in ${w.model} generations`}
                          >
                            {w.word} ({w.count})
                          </span>
                        ))}
                      </div>
                    ) : (
                      <div className="text-[9px] text-slate-400 font-mono italic">No automated signature words matched.</div>
                    )}
                  </div>
                </div>

                {/* 6. Citation Integrity Audit */}
                <div className="py-6">
                  <div className="flex items-center gap-1.5 text-md font-display font-medium text-slate-900 mb-4 tracking-wide uppercase">
                    <BookBookmark weight="duotone" className="w-4 h-4 text-[#7A2331]" />
                    Citation Integrity Audit
                  </div>
                  
                  <div className="p-3 bg-[#FAF8F2] border border-[#DCD4C0] rounded">
                    {/* Style format label */}
                    <div className="flex justify-between items-center mb-3 pb-2 border-b border-[#DCD4C0]/40 text-xs font-mono text-slate-600">
                      <span>DETECTED FORMAT:</span>
                      <span className="font-bold text-[#7A2331] uppercase">
                        {result.citation_audit.detected_style || 'UNKNOWN'}
                        {result.citation_audit.style_confidence ? ` (${(result.citation_audit.style_confidence * 100).toFixed(0)}% CONF)` : ''}
                      </span>
                    </div>

                    {result.citation_audit.detected_style === 'Unknown' && (
                      <div className="mb-3 p-2 bg-[#B23A2E]/5 border border-[#B23A2E]/20 text-[#B23A2E] text-xs rounded leading-normal font-mono">
                        Citation format not confidently detected — matching may be incomplete.
                      </div>
                    )}

                    <div className="flex justify-between items-center mb-2 font-mono text-slate-500">
                      <span className="text-sm">CITATION HEALTH:</span>
                      <span className={`font-mono text-lg font-bold ${result.citation_audit.health_score >= 80 ? 'text-[#4B6A57]' : result.citation_audit.health_score >= 50 ? 'text-[#B8862E]' : 'text-[#B23A2E]'}`}>
                        {result.citation_audit.health_score}%
                      </span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-200/50 rounded-full overflow-hidden mb-3">
                      <div 
                        className={`h-full transition-all duration-500 ease-out
                          ${result.citation_audit.health_score >= 80 ? 'bg-[#4B6A57]' :
                            result.citation_audit.health_score >= 50 ? 'bg-[#B8862E]' : 'bg-[#B23A2E]'}`}
                        style={{ width: `${result.citation_audit.health_score}%` }}
                      ></div>
                    </div>
                    
                    {/* Reference Audit List */}
                    {result.citation_audit.references.length > 0 ? (
                      <div className="space-y-[10px] mt-3 max-h-[220px] overflow-y-auto pr-1">
                        {result.citation_audit.references.map((ref, idx) => {
                          const isExpanded = expandedCitationIdx === idx;
                          
                          // Style-aware DOI verification statuses
                          const isVerified = ref.status === 'verified';
                          const isPartial = ref.status === 'partial_match';
                          const isMismatch = ref.status === 'mismatch';
                          const isDoiNotFound = ref.status === 'doi_not_found';
                          const isNoDoi = ref.status === 'no_doi_present';
                          const isLookupFailed = ref.status === 'lookup_failed';
                          
                          // Legacy compatibility
                          const isHallucinated = isMismatch || isDoiNotFound;

                          const key = getCitationKey(ref.reference);

                          // Check year mismatch
                          const bibYearMatch = ref.reference.match(/\b(19\d{2}|20\d{2})\b/);
                          const dbYearMatch = ref.details.match(/\b(19\d{2}|20\d{2})\b/);
                          const bibYear = bibYearMatch ? bibYearMatch[1] : null;
                          const dbYear = dbYearMatch ? dbYearMatch[1] : null;
                          const isYearMismatch = bibYear && dbYear && bibYear !== dbYear;

                          return (
                            <div key={idx} className="border-b border-[#DCD4C0]/40 pb-2.5 last:border-0 last:pb-0 transition-all">
                              {/* Row Header - Clickable for details */}
                              <div 
                                onClick={() => setExpandedCitationIdx(isExpanded ? null : idx)}
                                onMouseEnter={() => key && setHoveredCitationKey(key)}
                                onMouseLeave={() => setHoveredCitationKey(null)}
                                className="flex items-start gap-1.5 hover:bg-[#FAF8F2] p-1.5 rounded transition-colors select-none cursor-pointer"
                              >
                                {isVerified ? (
                                  <span className="text-[#4B6A57] font-bold text-sm shrink-0 mt-0.5" title="DOI Verified via CrossRef">✓</span>
                                ) : isPartial ? (
                                  <span className="text-[#B8862E] font-bold text-sm shrink-0 mt-0.5" title="Review Suggested: Stated reference partially matches metadata">⚠</span>
                                ) : isMismatch ? (
                                  <span className="text-[#B23A2E] font-bold text-sm shrink-0 mt-0.5 animate-pulse" title="Warning: Metadata mismatch. Stated citation title does not match registered title">⚠</span>
                                ) : isDoiNotFound ? (
                                  <span className="text-[#B23A2E] font-bold text-sm shrink-0 mt-0.5 animate-pulse" title="DOI could not be verified — may be fabricated or contain a typo">⚠</span>
                                ) : isLookupFailed ? (
                                  <span className="text-slate-400 font-bold text-sm shrink-0 mt-0.5" title="CrossRef verification couldn't be completed (network issue)">?</span>
                                ) : (
                                  <span className="text-slate-400 font-bold text-sm shrink-0 mt-0.5" title="No DOI present in reference entry">?</span>
                                )}
                                
                                <div className="flex-1 min-w-0">
                                  <span 
                                    className={`text-slate-700 text-base font-normal font-serif block leading-snug break-words ${!isExpanded ? 'truncate' : ''}`}
                                    title={ref.reference}
                                  >
                                    {ref.reference}
                                  </span>
                                  {ref.doi && (
                                    <a
                                      href={`https://doi.org/${ref.doi}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      onClick={(e) => e.stopPropagation()}
                                      className="inline-block text-[11px] font-mono text-[#7A2331] hover:underline mt-0.5"
                                    >
                                      DOI: {ref.doi} ↗
                                    </a>
                                  )}
                                  {!isExpanded && (
                                    <span className="text-[10px] font-mono text-slate-400 hover:text-[#7A2331] block mt-0.5">
                                      Click to audit details {key ? `(${key})` : ''} ⌐
                                    </span>
                                  )}
                                </div>

                                <span className="text-xs font-mono text-slate-300 select-none shrink-0 self-center pl-1">
                                  {isExpanded ? "[-]" : "[+]"}
                                </span>
                              </div>

                              {/* Expanded Detail Drawer */}
                              {isExpanded && (
                                <div className="mt-3 ml-[18px] p-3 bg-white border border-[#DCD4C0] rounded space-y-3 text-xs font-mono shadow-inner animate-fade-in">
                                  <div className="flex justify-between items-center pb-1.5 border-b border-[#DCD4C0]/30">
                                    <span className="text-slate-400 tracking-wider text-[9px] uppercase">AUDIT STATUS:</span>
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase
                                      ${isVerified ? 'bg-[#4B6A57]/10 text-[#4B6A57] border border-[#4B6A57]/20' : 
                                        isPartial ? 'bg-[#B8862E]/10 text-[#B8862E] border border-[#B8862E]/20' :
                                        isMismatch || isDoiNotFound ? 'bg-[#B23A2E]/10 text-[#B23A2E] border border-[#B23A2E]/20' : 
                                        'bg-slate-100 text-slate-600 border border-slate-200'}`}>
                                      {isVerified ? 'Verified Match' : 
                                       isPartial ? 'Review Suggested' : 
                                       isMismatch ? 'Title Mismatch' : 
                                       isDoiNotFound ? 'DOI Not Found' : 
                                       isLookupFailed ? 'Lookup Failed' : 
                                       'No DOI Present'}
                                    </span>
                                  </div>

                                  {/* Forensic Warnings */}
                                  {(isYearMismatch || isMismatch || isDoiNotFound || isPartial) && (
                                     <div className="space-y-1.5">
                                       <span className="text-[#7A2331] tracking-wider text-[9px] block font-bold uppercase">Forensic Alerts:</span>
                                       {isYearMismatch && (
                                         <div className="text-[10px] text-[#B8862E] bg-amber-50/50 border border-amber-200/50 rounded p-2 flex items-start gap-1.5 leading-normal">
                                           <span className="shrink-0 mt-0.5">⚠</span>
                                           <span>Year discrepancy: Bibliography cites {bibYear}, but database record shows {dbYear}. Common in generated or fabricated papers.</span>
                                         </div>
                                       )}
                                       {isMismatch && (
                                         <div className="text-[10px] text-[#B23A2E] bg-red-50/50 border border-red-200/50 rounded p-2 flex items-start gap-1.5 leading-normal">
                                           <span className="shrink-0 mt-0.5">☠</span>
                                           <span>Database verification mismatch: Stated reference title does not match DOI registry records. High fabrication or modification probability.</span>
                                         </div>
                                       )}
                                       {isDoiNotFound && (
                                         <div className="text-[10px] text-[#B23A2E] bg-red-50/50 border border-red-200/50 rounded p-2 flex items-start gap-1.5 leading-normal">
                                           <span className="shrink-0 mt-0.5">☠</span>
                                           <span>DOI not found: The DOI does not resolve to any registered publication. Likely fabricated or contains a typo.</span>
                                         </div>
                                       )}
                                       {isPartial && (
                                         <div className="text-[10px] text-[#B8862E] bg-amber-50/50 border border-amber-200/50 rounded p-2 flex items-start gap-1.5 leading-normal">
                                           <span className="shrink-0 mt-0.5">⚠</span>
                                           <span>Partial match: Reference title or metadata partially matches DOI registry records. Verify formatting.</span>
                                         </div>
                                       )}
                                     </div>
                                   )}

                                  <div className="space-y-1">
                                    <span className="text-slate-400 tracking-wider text-[9px] block uppercase">EXTRACTED BIBLIOGRAPHY:</span>
                                    <span className="text-slate-800 font-serif block text-sm leading-normal break-words whitespace-normal bg-[#FAF8F2] p-2 rounded border border-[#DCD4C0]/50">
                                      {ref.reference}
                                    </span>
                                  </div>

                                  <div className="space-y-1">
                                    <span className="text-slate-400 tracking-wider text-[9px] block uppercase">RESOLVED DATABASE MATCH:</span>
                                    <span className={`block p-2 rounded border leading-normal break-words whitespace-normal
                                      ${isVerified ? 'bg-slate-50/50 text-slate-800 border-[#4B6A57]/30' : 
                                        isHallucinated ? 'bg-[#B23A2E]/5 text-[#B23A2E] border-[#B23A2E]/20' : 
                                        'bg-slate-50 text-slate-600 border-slate-200'}`}>
                                      {ref.details}
                                    </span>
                                  </div>

                                  <div className="flex gap-2 pt-1 justify-end">
                                    <a 
                                      href={`https://scholar.google.com/scholar?q=${encodeURIComponent(getScholarQuery(ref.reference))}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="px-2.5 py-1 text-[10px] border border-[#DCD4C0] rounded hover:border-[#7A2331] hover:text-[#7A2331] bg-[#FAF8F2] hover:bg-white transition-all font-semibold uppercase"
                                    >
                                      Google Scholar ↗
                                    </a>
                                    <a 
                                      href={`https://api.crossref.org/works?query=${encodeURIComponent(ref.reference)}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="px-2.5 py-1 text-[10px] border border-[#DCD4C0] rounded hover:border-[#7A2331] hover:text-[#7A2331] bg-[#FAF8F2] hover:bg-white transition-all font-semibold uppercase"
                                    >
                                      CrossRef Record ↗
                                    </a>
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-sm text-slate-400 italic font-mono text-center py-2">
                        No bibliography sections matched.
                      </div>
                    )}

                    {/* Unmatched citations list */}
                    {result.citation_audit.unmatched_citations && result.citation_audit.unmatched_citations.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-[#DCD4C0]/40">
                        <span className="text-[10px] font-mono text-[#B23A2E] font-bold block mb-1.5 uppercase">
                          Unmatched In-text Citations ({result.citation_audit.unmatched_citations.length}):
                        </span>
                        <div className="space-y-1 max-h-[80px] overflow-y-auto pr-1">
                          {result.citation_audit.unmatched_citations.map((cit, idx) => (
                            <div key={idx} className="text-xs font-mono text-[#B23A2E] bg-red-50/50 p-1.5 rounded border border-[#B23A2E]/20 flex justify-between">
                              <span>Marker: <strong className="font-bold">{cit.marker_text}</strong></span>
                              <span>Target: {typeof cit.referenced_value === 'string' ? cit.referenced_value : JSON.stringify(cit.referenced_value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Unreferenced entries list */}
                    {result.citation_audit.unreferenced_entries && result.citation_audit.unreferenced_entries.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-[#DCD4C0]/40">
                        <span className="text-[10px] font-mono text-[#B8862E] font-bold block mb-1.5 uppercase">
                          Unreferenced References ({result.citation_audit.unreferenced_entries.length}):
                        </span>
                        <div className="space-y-1 max-h-[80px] overflow-y-auto pr-1">
                          {result.citation_audit.unreferenced_entries.map((ref, idx) => (
                            <div key={idx} className="text-[11px] font-mono text-slate-700 bg-amber-50/50 p-1.5 rounded border border-amber-200/20 truncate" title={ref.raw_text}>
                              Index {ref.entry_number_or_index}: {ref.raw_text}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 7. Summary Tally */}
                <div className="pt-6 font-mono text-slate-500 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">HIGH CONFIDENCE FLAGS:</span>
                    <span className="text-lg font-bold text-[#B23A2E]">{result.summary.high_confidence_count}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">MEDIUM CONFIDENCE FLAGS:</span>
                    <span className="text-lg font-bold text-[#B8862E]">{result.summary.medium_confidence_count}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">VERIFIED HUMAN PASSAGES:</span>
                    <span className="text-lg font-bold text-[#4B6A57]">{result.summary.unflagged_count}</span>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}
      </main>

    </div>
  );
}

// Framer motion variants
const sealVariants = {
  initial: { scale: 1.5, rotate: -20, opacity: 0 },
  animate: { 
    scale: 1, 
    rotate: 0, 
    opacity: 1,
    transition: { type: "spring" as any, stiffness: 350, damping: 20 }
  }
};

const shadowVariants = {
  initial: { boxShadow: "0 0 0 rgba(122, 35, 49, 0)" },
  animate: {
    boxShadow: [
      "0 0 0 0px rgba(122, 35, 49, 0)",
      "0 0 0 8px rgba(122, 35, 49, 0.12)",
      "0 0 0 16px rgba(122, 35, 49, 0)"
    ],
    transition: { delay: 0.15, duration: 0.45, ease: "easeOut" as any }
  }
};
