'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sliders, Sparkle, Info, ChartBar, CheckCircle } from '@phosphor-icons/react';

interface StylometricGaugesProps {
  lexicalDiversity?: number | string;
  structuralBurstiness?: number | string;
  totalSentences?: number;
}

interface ExecutiveMetric {
  id: string;
  title: string;
  value: number; // Strictly 0 - 100%
  statusLabel: string;
  statusColor: string; // Tailored status color
  strokeGradient: [string, string];
  description: string;
  benchmarks: string;
}

/**
 * Professional Executive Semi-Circular Radial Gauge
 */
const ProfessionalRadialGauge: React.FC<{
  metric: ExecutiveMetric;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ metric, isSelected, onSelect }) => {
  const [isHovered, setIsHovered] = useState(false);

  // Clamp percentage between 0 and 100
  const percentage = Math.min(100, Math.max(0, metric.value));

  // Semi-circle arc parameters (Radius = 38, Center = 50,50)
  const radius = 38;
  // Semi-circle circumference = π * R
  const arcCircumference = Math.PI * radius; 
  const strokeOffset = arcCircumference - (arcCircumference * percentage) / 100;

  const gradientId = `prof-grad-${metric.id}`;

  return (
    <motion.div
      whileHover={{ y: -2 }}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`p-4 rounded-xl transition-all cursor-pointer relative overflow-hidden flex flex-col items-center justify-between border ${
        isSelected
          ? 'bg-slate-900/90 border-indigo-500/60 shadow-lg'
          : 'bg-slate-950/70 border-slate-800/90 hover:border-slate-700 hover:bg-slate-900/60'
      }`}
    >
      {/* Header */}
      <div className="w-full flex items-center justify-between mb-1 text-xs font-semibold">
        <span className="text-slate-300 font-medium truncate">{metric.title}</span>
        <span 
          className="text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase tracking-wider"
          style={{ backgroundColor: `${metric.statusColor}18`, color: metric.statusColor, border: `1px solid ${metric.statusColor}40` }}
        >
          {metric.statusLabel}
        </span>
      </div>

      {/* Professional Semi-Circular Arc Meter */}
      <div className="relative w-36 h-22 flex items-center justify-center my-1">
        <svg viewBox="0 0 100 55" className="w-full h-full overflow-visible">
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={metric.strokeGradient[0]} />
              <stop offset="100%" stopColor={metric.strokeGradient[1]} />
            </linearGradient>
          </defs>

          {/* Background Arc Track */}
          <path
            d="M 12 50 A 38 38 0 0 1 88 50"
            fill="none"
            stroke="#1e293b"
            strokeWidth="7"
            strokeLinecap="round"
          />

          {/* Value Arc Sweep */}
          <path
            d="M 12 50 A 38 38 0 0 1 88 50"
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth="7"
            strokeDasharray={arcCircumference}
            strokeDashoffset={strokeOffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        {/* Center Digital Display */}
        <div className="absolute bottom-1 flex flex-col items-center justify-center text-center">
          <span className="text-2xl font-bold font-mono tracking-tight text-white">
            {percentage.toFixed(1)}
            <span className="text-xs text-slate-400 font-normal ml-0.5">%</span>
          </span>
          <span className="text-[10px] text-slate-400 font-medium">100% Scale</span>
        </div>
      </div>

      {/* Bottom Track Scale Indicator */}
      <div className="w-full mt-1 pt-1.5 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-400">
        <span>0%</span>
        <span className="text-slate-300 font-semibold">{percentage.toFixed(1)}% Score</span>
        <span>100%</span>
      </div>
    </motion.div>
  );
};

export const StylometricGauges: React.FC<StylometricGaugesProps> = ({
  lexicalDiversity = 74.2,
  structuralBurstiness = 8.4,
  totalSentences = 0
}) => {
  // Parse input metrics
  const lexNum = typeof lexicalDiversity === 'number'
    ? lexicalDiversity
    : parseFloat(String(lexicalDiversity)) || 74.2;

  const burstRaw = typeof structuralBurstiness === 'number'
    ? structuralBurstiness
    : parseFloat(String(structuralBurstiness)) || 8.4;

  // Calibrate every metric strictly to a 0% - 100% scale
  const lexVal = Math.min(100, Math.max(0, lexNum));
  const burstVal = Math.min(100, Math.max(0, (burstRaw / 35) * 100));

  const metrics: ExecutiveMetric[] = [
    {
      id: 'lexical',
      title: 'Lexical Diversity (TTR)',
      value: lexVal,
      statusLabel: lexVal >= 75 ? 'Optimal' : lexVal >= 50 ? 'Standard' : 'Repetitive',
      statusColor: lexVal >= 75 ? '#10b981' : lexVal >= 50 ? '#6366f1' : '#f59e0b',
      strokeGradient: ['#6366f1', '#10b981'],
      description: 'Percentage of unique vocabulary tokens relative to total word count (Type-Token Ratio).',
      benchmarks: 'Human academic papers typically score 70% – 95%.'
    },
    {
      id: 'burstiness',
      title: 'Structural Burstiness',
      value: burstVal,
      statusLabel: burstVal >= 55 ? 'High Variance' : burstVal >= 30 ? 'Moderate' : 'Uniform',
      statusColor: burstVal >= 55 ? '#10b981' : burstVal >= 30 ? '#6366f1' : '#ef4444',
      strokeGradient: ['#3b82f6', '#06b6d4'],
      description: 'Sentence length variance index measuring natural stylistic shifts in paragraph cadence.',
      benchmarks: 'High human variation > 50%; uniform AI rhythm < 30%.'
    }
  ];

  const [selectedId, setSelectedId] = useState<string>('lexical');
  const activeMetric = metrics.find((m) => m.id === selectedId) || metrics[0];

  return (
    <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800/90 space-y-3 shadow-xl backdrop-blur-sm">
      {/* Executive Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Sliders className="w-4 h-4 text-indigo-400" />
          Stylometric Entropy Gauges
        </h4>
        <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800 flex items-center gap-1">
          <CheckCircle className="w-3 h-3 text-emerald-400" />
          100% Calibrated
        </span>
      </div>

      {/* 2 Side-by-Side Professional Radial Arc Gauges */}
      <div className="grid grid-cols-2 gap-3">
        {metrics.map((m) => (
          <ProfessionalRadialGauge
            key={m.id}
            metric={m}
            isSelected={m.id === selectedId}
            onSelect={() => setSelectedId(m.id)}
          />
        ))}
      </div>

      {/* Selected Metric Executive Details */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeMetric.id}
          initial={{ opacity: 0, y: 3 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -3 }}
          className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs space-y-1"
        >
          <div className="flex items-center justify-between text-slate-200 font-semibold">
            <span className="flex items-center gap-1.5 text-indigo-300">
              <Info className="w-3.5 h-3.5 text-indigo-400" />
              {activeMetric.title} Analysis
            </span>
            <span className="font-mono text-emerald-400 font-bold">{activeMetric.value.toFixed(1)}% / 100%</span>
          </div>
          <p className="text-slate-400 text-[11px] leading-relaxed">{activeMetric.description}</p>
          <p className="text-[10px] text-slate-400 font-mono pt-1 border-t border-slate-900">
            Benchmark: <span className="text-slate-300">{activeMetric.benchmarks}</span>
          </p>
        </motion.div>
      </AnimatePresence>

      {/* Footer Metadata */}
      <div className="flex items-center justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800 text-xs">
        <span className="text-slate-400">Total Analyzed Sentences:</span>
        <span className="font-mono font-bold text-slate-200">{totalSentences}</span>
      </div>
    </div>
  );
};
