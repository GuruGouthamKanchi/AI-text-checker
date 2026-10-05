'use client';

import React, { useEffect, useState } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

export const CustomCursor: React.FC = () => {
  const mouseX = useMotionValue(-100);
  const mouseY = useMotionValue(-100);

  // Smooth trailing spring physics for outer reticle ring
  const reticleX = useSpring(mouseX, { stiffness: 700, damping: 32 });
  const reticleY = useSpring(mouseY, { stiffness: 700, damping: 32 });

  const [isHovered, setIsHovered] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [hoverText, setHoverText] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isOverSelect, setIsOverSelect] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    let rafId: number | null = null;

    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
      if (!isVisible) setIsVisible(true);

      // Throttled DOM element check via requestAnimationFrame to avoid main-thread lock
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = null;
          const target = e.target as HTMLElement | null;
          if (target) {
            const isSelect = !!target.closest('select, option');
            setIsOverSelect(isSelect);

            const isInteractive = !!target.closest(
              'button, a, input, textarea, select, [role="button"], .cursor-pointer, [onClick]'
            );
            setIsHovered(isInteractive);

            const aiSentence = target.closest('[data-ai-percent]') as HTMLElement | null;
            if (aiSentence) {
              setHoverText(`${aiSentence.getAttribute('data-ai-percent')}% AI`);
            } else {
              setHoverText(null);
            }
          }
        });
      }
    };

    const handleMouseDown = () => setIsClicked(true);
    const handleMouseUp = () => setIsClicked(false);
    const handleMouseLeave = () => setIsVisible(false);
    const handleMouseEnter = () => setIsVisible(true);

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [isVisible, mouseX, mouseY]);

  if (!isVisible || isOverSelect) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
      {/* 1. Hardware-Accelerated 0-Latency Pointer */}
      <motion.div
        className="fixed top-0 left-0 w-7 h-7 z-50 pointer-events-none flex items-center justify-center origin-top-left"
        style={{
          x: mouseX,
          y: mouseY,
        }}
        animate={{
          scale: isClicked ? 0.8 : isHovered ? 1.25 : 1,
          rotate: isHovered ? -10 : 0,
        }}
        transition={{ type: 'spring', stiffness: 800, damping: 35 }}
      >
        <img
          src="/cursor.svg"
          alt="AI Cyber Forensic Cursor"
          className="w-full h-full drop-shadow-[0_2px_12px_rgba(99,102,241,0.6)]"
        />
      </motion.div>

      {/* 2. Spring-physics Trailing Reticle Ring */}
      <motion.div
        className="fixed top-0 left-0 rounded-full border border-indigo-400/60 shadow-2xl -translate-x-1/2 -translate-y-1/2 flex items-center justify-center z-40 backdrop-blur-[1px]"
        style={{
          x: reticleX,
          y: reticleY,
          width: isHovered ? '28px' : '18px',
          height: isHovered ? '28px' : '18px',
          borderColor: isHovered ? 'rgba(16, 185, 129, 0.8)' : 'rgba(99, 102, 241, 0.6)',
          backgroundColor: isHovered ? 'rgba(16, 185, 129, 0.1)' : 'rgba(99, 102, 241, 0.05)',
        }}
        animate={{
          scale: isClicked ? 0.8 : 1,
          rotate: 360,
        }}
        transition={{
          rotate: { duration: 10, repeat: Infinity, ease: 'linear' },
          scale: { type: 'spring', stiffness: 500, damping: 30 },
          width: { type: 'spring', stiffness: 500, damping: 30 },
          height: { type: 'spring', stiffness: 500, damping: 30 },
        }}
      >
        {isHovered && (
          <>
            <div className="absolute top-0 w-0.5 h-1 bg-emerald-400 rounded-full" />
            <div className="absolute bottom-0 w-0.5 h-1 bg-emerald-400 rounded-full" />
            <div className="absolute left-0 h-0.5 w-1 bg-emerald-400 rounded-full" />
            <div className="absolute right-0 h-0.5 w-1 bg-emerald-400 rounded-full" />
          </>
        )}
      </motion.div>

      {/* 3. Click Ripple Wave */}
      {isClicked && (
        <motion.div
          className="fixed top-0 left-0 rounded-full border-2 border-indigo-400/80 -translate-x-1/2 -translate-y-1/2 z-30"
          style={{
            x: mouseX,
            y: mouseY,
          }}
          initial={{ width: 6, height: 6, opacity: 1 }}
          animate={{ width: 36, height: 36, opacity: 0 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
        />
      )}

      {/* 4. AI Inspection Tooltip Badge */}
      {hoverText && (
        <motion.div
          className="fixed top-0 left-0 px-2 py-0.5 rounded bg-slate-900 border border-indigo-500/80 text-[10px] font-mono font-bold text-indigo-300 shadow-xl z-50 pointer-events-none"
          style={{
            x: mouseX,
            y: mouseY,
            translateX: 16,
            translateY: 16,
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          {hoverText}
        </motion.div>
      )}
    </div>
  );
};

export default CustomCursor;

