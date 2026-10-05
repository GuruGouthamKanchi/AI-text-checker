import { prepareWithSegments, layoutWithLines, measureLineStats, PreparedTextWithSegments } from '@chenglou/pretext';

export interface PretextParagraphLayout {
  paragraphIndex: number;
  linesCount: number;
  height: number;
  sampleLine: string;
}

export interface PretextMetrics {
  totalLines: number;
  totalHeight: number;
  maxLineWidth: number;
  computeTimeMs: number;
  fontUsed: string;
  paragraphLayouts: PretextParagraphLayout[];
}

/**
 * Analyzes document paragraphs off-screen using @chenglou/pretext.
 * Measures text layout, line breaks, heights, and max line widths
 * with 0 browser layout reflows (300-600x faster than DOM calls).
 */
export function analyzeTextWithPretext(
  paragraphs: string[],
  containerWidth: number = 750,
  fontSize: number = 15,
  lineHeight: number = 24
): PretextMetrics {
  const startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const font = `${fontSize}px Inter, sans-serif`;
  
  let totalLines = 0;
  let totalHeight = 0;
  let maxLineWidth = 0;

  const paragraphLayouts: PretextParagraphLayout[] = paragraphs.map((text, idx) => {
    if (!text || text.trim().length === 0) {
      return { paragraphIndex: idx, linesCount: 0, height: 0, sampleLine: '' };
    }

    try {
      const prepared: PreparedTextWithSegments = prepareWithSegments(text, font);
      const layoutResult = layoutWithLines(prepared, containerWidth, lineHeight);
      const lineStats = measureLineStats(prepared, containerWidth);

      totalLines += layoutResult.lineCount;
      totalHeight += layoutResult.height;
      if (lineStats.maxLineWidth > maxLineWidth) {
        maxLineWidth = lineStats.maxLineWidth;
      }

      const sampleLine = layoutResult.lines && layoutResult.lines.length > 0 
        ? layoutResult.lines[0].text 
        : text.slice(0, 40);

      return {
        paragraphIndex: idx,
        linesCount: layoutResult.lineCount,
        height: layoutResult.height,
        sampleLine
      };
    } catch {
      return { paragraphIndex: idx, linesCount: 1, height: lineHeight, sampleLine: text.slice(0, 40) };
    }
  });

  const endTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const computeTimeMs = Math.max(0.01, Math.round((endTime - startTime) * 100) / 100);

  return {
    totalLines,
    totalHeight,
    maxLineWidth: Math.round(maxLineWidth),
    computeTimeMs,
    fontUsed: font,
    paragraphLayouts
  };
}
