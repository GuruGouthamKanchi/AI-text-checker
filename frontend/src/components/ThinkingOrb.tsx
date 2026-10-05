'use client';

import React from 'react';
import { motion } from 'framer-motion';

export type OrbState = 'connecting' | 'thinking' | 'analyzing' | 'idle' | 'complete' | 'error';

export interface ThinkingOrbProps {
  state?: OrbState;
  size?: number;
  className?: string;
}

export const ThinkingOrb: React.FC<ThinkingOrbProps> = ({
  state = 'thinking',
  size = 64,
  className = ''
}) => {
  // State color mapping
  const getStateColors = () => {
    switch (state) {
      case 'connecting':
      case 'thinking':
        return {
          primary: '#6366f1', // Indigo
          secondary: '#a855f7', // Purple
          accent: '#38bdf8', // Cyan
          glow: 'rgba(99, 102, 241, 0.4)'
        };
      case 'analyzing':
        return {
          primary: '#10b981', // Emerald
          secondary: '#6366f1', // Indigo
          accent: '#34d399', // Mint
          glow: 'rgba(16, 185, 129, 0.4)'
        };
      case 'complete':
        return {
          primary: '#10b981', // Emerald
          secondary: '#059669',
          accent: '#6ee7b7',
          glow: 'rgba(16, 185, 129, 0.5)'
        };
      case 'error':
        return {
          primary: '#ef4444', // Red
          secondary: '#f43f5e', // Rose
          accent: '#fca5a5',
          glow: 'rgba(239, 68, 68, 0.5)'
        };
      case 'idle':
      default:
        return {
          primary: '#64748b', // Slate
          secondary: '#475569',
          accent: '#94a3b8',
          glow: 'rgba(100, 116, 139, 0.3)'
        };
    }
  };

  const colors = getStateColors();

  return (
    <div 
      className={`relative flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
    >
      {/* Outer Glowing Background Blur */}
      <motion.div
        className="absolute inset-0 rounded-full blur-xl pointer-events-none"
        animate={{
          scale: [1, 1.25, 1],
          opacity: [0.5, 0.8, 0.5],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        style={{
          background: `radial-gradient(circle, ${colors.primary} 0%, ${colors.secondary} 70%, transparent 100%)`,
        }}
      />

      {/* Outer Rotating Orbital Ring 1 */}
      <motion.div
        className="absolute rounded-full border border-dashed pointer-events-none"
        style={{
          width: size * 0.95,
          height: size * 0.95,
          borderColor: colors.accent,
          opacity: 0.6,
        }}
        animate={{ rotate: 360 }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      {/* Inner Reverse Rotating Orbital Ring 2 */}
      <motion.div
        className="absolute rounded-full border pointer-events-none"
        style={{
          width: size * 0.75,
          height: size * 0.75,
          borderColor: colors.secondary,
          borderTopColor: 'transparent',
          borderRightColor: 'transparent',
          opacity: 0.8,
          borderWidth: 2,
        }}
        animate={{ rotate: -360 }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: 'linear',
        }}
      />

      {/* Core Glowing Orb Sphere */}
      <motion.div
        className="relative rounded-full shadow-2xl flex items-center justify-center overflow-hidden"
        style={{
          width: size * 0.55,
          height: size * 0.55,
          background: `radial-gradient(circle at 35% 35%, ${colors.accent}, ${colors.primary} 60%, ${colors.secondary} 100%)`,
          boxShadow: `0 0 25px ${colors.glow}, inset 0 0 15px rgba(255, 255, 255, 0.6)`,
        }}
        animate={{
          scale: [0.95, 1.08, 0.95],
        }}
        transition={{
          duration: 2.2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        {/* Internal Shimmer Wave */}
        <motion.div
          className="absolute inset-0 bg-white/20 blur-sm rounded-full"
          animate={{
            x: ['-100%', '100%'],
            y: ['-100%', '100%'],
          }}
          transition={{
            duration: 2.5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </motion.div>
    </div>
  );
};

export default ThinkingOrb;
