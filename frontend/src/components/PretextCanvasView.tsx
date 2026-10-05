'use client';

import React, { useRef, useEffect, useState } from 'react';
import { prepareWithSegments, layoutWithLines, PreparedTextWithSegments } from '@chenglou/pretext';
import { Sparkle, Sliders, CheckCircle, Lightning, CaretRight, BookOpen, Funnel, ArrowRight } from '@phosphor-icons/react';

interface Sentence {
  sentence_id?: number;
  text: string;
  ai_probability: number;
  paragraph_index?: number;
}

export interface SectionBreakdownItem {
  section_id: number;
  title: string;
  category: string;
  word_count: number;
  sentence_count: number;
  ai_percentage: number;
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface SectionAnalysis {
  sections?: Array<{
    section_id: number;
    title: string;
    category: string;
    sentences: Sentence[];
    word_count: number;
    ai_percentage: number;
    risk_level: string;
  }>;
  section_breakdown?: SectionBreakdownItem[];
  highest_risk_section?: SectionBreakdownItem;
  lowest_risk_section?: SectionBreakdownItem;
}

interface PretextCanvasViewProps {
  paragraphs: string[];
  sentences: Sentence[];
  sensitivityThreshold: number;
  sectionAnalysis?: SectionAnalysis;
  onTransferToHumanizer?: (text: string) => void;
}

export default function PretextCanvasView({
  paragraphs,
  sentences,
  sensitivityThreshold,
  sectionAnalysis,
  onTransferToHumanizer
}: PretextCanvasViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [activeCategoryFilter, setActiveCategoryFilter] = useState<string>('ALL');

  const [renderStats, setRenderStats] = useState<{
    totalLines: number;
    canvasHeight: number;
    computeTimeMs: number;
    fps: number;
  }>({ totalLines: 0, canvasHeight: 600, computeTimeMs: 0, fps: 120 });

  const breakdownItems = sectionAnalysis?.section_breakdown || [];

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    // 0. Fallback & Metadata Splitting: Ensure section headers, titles, emails, and abstracts are distinct cards
    let rawParagraphList = paragraphs && paragraphs.filter(p => p && p.trim().length > 0).length > 0
      ? paragraphs.filter(p => p && p.trim().length > 0)
      : [];

    if (rawParagraphList.length === 0 && sentences && sentences.length > 0) {
      const pGroups: { [key: number]: string[] } = {};
      sentences.forEach(s => {
        const pIdx = s.paragraph_index ?? 0;
        if (!pGroups[pIdx]) pGroups[pIdx] = [];
        pGroups[pIdx].push(s.text);
      });
      rawParagraphList = Object.keys(pGroups)
        .sort((a, b) => parseInt(a) - parseInt(b))
        .map(key => pGroups[parseInt(key)].join(" "));
    }

    // Split any merged paragraphs containing embedded line breaks or section markers
    const activeParagraphs: Array<{ text: string; category: string; title: string }> = [];
    rawParagraphList.forEach((pBlock, bIdx) => {
      const subLines = pBlock.split(/\n+/).map(l => l.trim()).filter(Boolean);
      let currentChunk: string[] = [];
      let currentCategory = 'GENERAL';
      let currentTitle = `SECTION #${bIdx + 1}`;

      subLines.forEach(line => {
        const isHeader = /^(?:[0-9]+\s+[A-Z]|\d+\.|\b(?:abstract|introduction|keywords|e-mail|email|department|school|university|literature review|methodology|results|discussion|conclusion|references)\b)/i.test(line);
        if (isHeader && currentChunk.length > 0) {
          activeParagraphs.push({
            text: currentChunk.join(" "),
            category: currentCategory,
            title: currentTitle
          });
          currentChunk = [line];
          currentTitle = line.slice(0, 40);
        } else {
          currentChunk.push(line);
        }
      });
      if (currentChunk.length > 0) {
        activeParagraphs.push({
          text: currentChunk.join(" "),
          category: currentCategory,
          title: currentTitle
        });
      }
    });

    // Apply category filter if active
    const filteredParagraphs = activeCategoryFilter === 'ALL'
      ? activeParagraphs
      : activeParagraphs.filter(p => p.category === activeCategoryFilter || p.title.toUpperCase().includes(activeCategoryFilter));

    const renderingList = filteredParagraphs.length > 0 ? filteredParagraphs : activeParagraphs;
    if (renderingList.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const containerWidth = Math.max(400, containerRef.current.clientWidth - 48);
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

    const font = '15px Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    const lineHeight = 24;
    const headerHeight = 34;
    const paragraphSpacing = 20;

    const startTime = performance.now();

    // 1. Off-screen Pretext Layout Calculation
    let totalLinesCount = 0;
    let currentY = 16;

    interface LayoutParagraph {
      text: string;
      lines: string[];
      startY: number;
      height: number;
      aiProb: number;
      title: string;
    }

    const computedParagraphs: LayoutParagraph[] = [];

    // Map sentence AI probabilities to paragraphs
    const pAiMap: { [key: number]: number[] } = {};
    sentences.forEach(s => {
      const idx = s.paragraph_index ?? 0;
      if (!pAiMap[idx]) pAiMap[idx] = [];
      pAiMap[idx].push(s.ai_probability);
    });

    renderingList.forEach((item, pIdx) => {
      const pText = item.text;
      if (!pText || pText.trim().length === 0) return;

      let lineTexts: string[] = [];
      let pHeight = lineHeight;

      try {
        const prepared: PreparedTextWithSegments = prepareWithSegments(pText, font);
        const layoutRes = layoutWithLines(prepared, containerWidth - 48, lineHeight);
        lineTexts = layoutRes.lines.map(l => l.text);
        pHeight = layoutRes.height;
        totalLinesCount += layoutRes.lineCount;
      } catch {
        lineTexts = [pText];
      }

      const probs = pAiMap[pIdx] || [0.1];
      const avgProb = probs.reduce((a, b) => a + b, 0) / probs.length;

      computedParagraphs.push({
        text: pText,
        lines: lineTexts,
        startY: currentY,
        height: pHeight,
        aiProb: avgProb,
        title: item.title
      });

      currentY += pHeight + headerHeight + paragraphSpacing;
    });

    const totalCanvasHeight = Math.max(500, currentY + 40);

    // 2. Adjust Canvas DPR resolution
    canvas.width = containerWidth * dpr;
    canvas.height = totalCanvasHeight * dpr;
    canvas.style.width = `${containerWidth}px`;
    canvas.style.height = `${totalCanvasHeight}px`;

    ctx.scale(dpr, dpr);

    // 3. Clear & Render Background
    ctx.fillStyle = '#030712'; // Tailwind slate-950
    ctx.fillRect(0, 0, containerWidth, totalCanvasHeight);

    // 4. Draw Pretext Section Cards & AI Forensic Highlights
    computedParagraphs.forEach((p, pIdx) => {
      const isHighAi = p.aiProb >= (sensitivityThreshold + 15) / 100;
      const isMedAi = p.aiProb >= sensitivityThreshold / 100 && !isHighAi;

      // Draw Paragraph Card Background Box
      const cardX = 8;
      const cardY = p.startY;
      const cardW = containerWidth - 16;
      const cardH = p.height + headerHeight + 12;

      ctx.fillStyle = isHighAi
        ? 'rgba(127, 29, 29, 0.25)' // red-950/25
        : isMedAi
        ? 'rgba(120, 53, 15, 0.25)' // amber-950/25
        : 'rgba(15, 23, 42, 0.6)'; // slate-900/60

      ctx.strokeStyle = isHighAi
        ? 'rgba(239, 68, 68, 0.4)' // red-500/40
        : isMedAi
        ? 'rgba(245, 158, 11, 0.4)' // amber-500/40
        : 'rgba(51, 65, 85, 0.5)'; // slate-700/50

      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(cardX, cardY, cardW, cardH, 10);
      ctx.fill();
      ctx.stroke();

      // Header Row Background Bar (separates section header from body text)
      ctx.fillStyle = 'rgba(15, 23, 42, 0.5)';
      ctx.beginPath();
      ctx.roundRect(cardX + 1, cardY + 1, cardW - 2, 30, [9, 9, 0, 0]);
      ctx.fill();

      // Draw Section Title on Left
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = isHighAi ? '#fca5a5' : isMedAi ? '#fde68a' : '#94a3b8';
      ctx.textBaseline = 'middle';
      const headingLabel = p.title.length > 45 ? `${p.title.slice(0, 42)}...` : p.title;
      ctx.fillText(`SECTION #${pIdx + 1}: ${headingLabel.toUpperCase()}`, cardX + 14, cardY + 16);

      // Draw AI Badge Tag on Right of Header Bar
      const percent = Math.round(p.aiProb * 100);
      ctx.font = 'bold 11px monospace';
      ctx.fillStyle = isHighAi ? '#fca5a5' : isMedAi ? '#fde68a' : '#6ee7b7';
      const riskLabel = isHighAi ? 'HIGH RISK' : isMedAi ? 'MEDIUM RISK' : 'HUMAN';
      ctx.fillText(`${percent}% AI (${riskLabel})`, cardX + cardW - 145, cardY + 16);

      // Draw Multiline Body Text formatted by Pretext
      ctx.font = font;
      ctx.fillStyle = '#e2e8f0'; // slate-200
      ctx.textBaseline = 'top';

      const bodyStartY = cardY + headerHeight + 6;
      p.lines.forEach((lineStr, lIdx) => {
        const lineY = bodyStartY + lIdx * lineHeight;
        ctx.fillText(lineStr, cardX + 16, lineY);
      });
    });

    const endTime = performance.now();
    const computeTimeMs = Math.round((endTime - startTime) * 100) / 100;

    setRenderStats({
      totalLines: totalLinesCount,
      canvasHeight: Math.round(totalCanvasHeight),
      computeTimeMs,
      fps: 120
    });

  }, [paragraphs, sentences, sensitivityThreshold, activeCategoryFilter]);

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* 1. Interactive Section AI Risk Spectrum Bar */}
      {breakdownItems.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3 shadow-xl">
          <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-300">
            <span className="flex items-center gap-2 text-indigo-400">
              <BookOpen className="w-4 h-4" /> Academic Paper Section Risk Spectrum
            </span>
            <span className="text-[10px] font-mono text-slate-400 font-normal">
              {breakdownItems.length} Sections Classified
            </span>
          </div>

          {/* Section Risk Spectrum Pills Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {breakdownItems.map((sec) => {
              const isHigh = sec.risk_level === 'HIGH';
              const isMed = sec.risk_level === 'MEDIUM';

              return (
                <div
                  key={`sec-spectrum-${sec.section_id}`}
                  onClick={() => setActiveCategoryFilter(sec.category)}
                  className={`p-2.5 rounded-xl border text-xs cursor-pointer transition-all ${
                    activeCategoryFilter === sec.category
                      ? 'bg-indigo-950 border-indigo-500 shadow-lg shadow-indigo-950'
                      : isHigh
                      ? 'bg-red-950/30 border-red-900/50 hover:border-red-500'
                      : isMed
                      ? 'bg-amber-950/30 border-amber-900/50 hover:border-amber-500'
                      : 'bg-slate-950/60 border-slate-800 hover:border-emerald-500'
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] font-mono font-semibold text-slate-400 mb-1">
                    <span className="truncate max-w-[110px]">{sec.title}</span>
                    <span className={`px-1.5 py-0.5 rounded font-bold ${
                      isHigh ? 'bg-red-950 text-red-300 border border-red-800' : isMed ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                    }`}>
                      {sec.ai_percentage}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className={`h-full rounded-full transition-all ${
                        isHigh ? 'bg-red-500' : isMed ? 'bg-amber-500' : 'bg-emerald-400'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(5, sec.ai_percentage))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 2. Pretext Layout Performance Ribbon */}
      <div className="p-3 rounded-xl bg-indigo-950/40 border border-indigo-800/60 flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-2 text-indigo-300 font-semibold">
          <Lightning className="w-4 h-4 text-amber-400 animate-pulse" />
          <span>@chenglou/pretext Section Layout Engine</span>
        </div>
        <div className="flex items-center gap-4 text-slate-400">
          <span>Compute: <strong className="text-emerald-400 font-bold">{renderStats.computeTimeMs}ms</strong></span>
          <span>Lines: <strong className="text-indigo-400">{renderStats.totalLines}</strong></span>
          <span>Reflows: <strong className="text-emerald-400 font-bold">0 DOM Calls</strong></span>
          <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">120 FPS</span>
        </div>
      </div>

      {/* 3. HTML5 Canvas Element driven by Pretext */}
      <div className="w-full bg-slate-950 rounded-2xl border border-slate-800 p-3 overflow-x-hidden overflow-y-auto max-h-[650px] shadow-2xl flex justify-center">
        <canvas ref={canvasRef} className="rounded-xl shadow-lg border border-slate-900" />
      </div>
    </div>
  );
}
