/**
 * Reference Intelligence & Multi-Format Citation Generator Utilities
 * Provides title extraction, Google Scholar / Semantic Scholar URL generation,
 * DOI/arXiv detection, and BibTeX, RIS, APA 7th, and IEEE citation formatting.
 */

export interface ExtractedReferenceInfo {
  cleanTitle: string;
  authors: string;
  year?: string;
  doi?: string;
  arxivId?: string;
  journal?: string;
}

/**
 * Extracts clean paper title, authors, year, DOI, and arXiv IDs from reference text.
 */
export function parseReferenceText(referenceText: string): ExtractedReferenceInfo {
  let text = referenceText.replace(/^\[\d+\]\s*/, '').trim();

  // Extract DOI if present
  const doiMatch = text.match(/\b10\.\d{4,9}\/[-._;()/:A-Za-z0-9]+/);
  let doi: string | undefined = doiMatch ? doiMatch[0] : undefined;
  if (doi) {
    while (doi && ".,;)]}".includes(doi.slice(-1))) {
      doi = doi.slice(0, -1);
    }
  }

  // Extract arXiv ID if present
  const arxivMatch = text.match(/arXiv:(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]+)?\/\d{7})/i);
  const arxivId: string | undefined = arxivMatch ? arxivMatch[1] : undefined;

  // Extract Year
  const yearMatch = text.match(/\b(19\d{2}|20\d{2})\b/);
  const year: string | undefined = yearMatch ? yearMatch[1] : undefined;

  // Extract Title inside quotes ("Title" or “Title”)
  const quoteMatch = text.match(/["“]([^"”]+)["”]/);
  let cleanTitle = "";
  let authors = "";

  if (quoteMatch && quoteMatch[1].length > 5) {
    cleanTitle = quoteMatch[1].trim();
    const titleIdx = text.indexOf(quoteMatch[0]);
    if (titleIdx > 0) {
      authors = text.substring(0, titleIdx).replace(/[,;]\s*$/, '').trim();
    }
  } else {
    const parts = text.split('.');
    if (parts.length >= 2 && parts[0].length < 100) {
      authors = parts[0].trim();
      cleanTitle = parts[1].trim();
    } else {
      cleanTitle = text;
    }
  }

  cleanTitle = cleanTitle.replace(/[,;.]\s*$/, '').trim();

  return {
    cleanTitle: cleanTitle || text,
    authors: authors || "Unknown Authors",
    year,
    doi,
    arxivId
  };
}

/**
 * Generates Google Scholar search URL for a reference.
 */
export function getGoogleScholarUrl(referenceText: string): string {
  const info = parseReferenceText(referenceText);
  const query = info.cleanTitle.length > 10 ? info.cleanTitle : referenceText;
  return `https://scholar.google.com/scholar?q=${encodeURIComponent(query)}`;
}

/**
 * Generates Semantic Scholar search URL for a reference.
 */
export function getSemanticScholarUrl(referenceText: string): string {
  const info = parseReferenceText(referenceText);
  const query = info.cleanTitle.length > 10 ? info.cleanTitle : referenceText;
  return `https://www.semanticscholar.org/search?q=${encodeURIComponent(query)}`;
}

/**
 * Formats reference info into standard BibTeX entry string.
 */
export function generateBibTeX(referenceText: string, index: number = 1): string {
  const info = parseReferenceText(referenceText);
  const firstAuthorLastName = info.authors.split(',')[0].split(' ').pop()?.toLowerCase() || 'ref';
  const citeKey = `${firstAuthorLastName}${info.year || '2024'}_${index}`;

  let bib = `@article{${citeKey},\n`;
  bib += `  title = {${info.cleanTitle}},\n`;
  if (info.authors) bib += `  author = {${info.authors}},\n`;
  if (info.year) bib += `  year = {${info.year}},\n`;
  if (info.doi) bib += `  doi = {${info.doi}},\n`;
  if (info.arxivId) bib += `  eprint = {${info.arxivId}},\n`;
  bib += `  note = {Audited by VeriPaper AI}\n`;
  bib += `}`;
  
  return bib;
}

/**
 * Formats reference info into Research Information Systems (RIS) format for Zotero/EndNote.
 */
export function generateRIS(referenceText: string): string {
  const info = parseReferenceText(referenceText);
  let ris = "TY  - JOUR\n";
  ris += `TI  - ${info.cleanTitle}\n`;
  if (info.authors) ris += `AU  - ${info.authors}\n`;
  if (info.year) ris += `PY  - ${info.year}\n`;
  if (info.doi) ris += `DO  - ${info.doi}\n`;
  if (info.arxivId) ris += `UR  - https://arxiv.org/abs/${info.arxivId}\n`;
  ris += "ER  - \n";
  return ris;
}

/**
 * Formats reference info into APA 7th Edition style.
 */
export function generateAPA(referenceText: string): string {
  const info = parseReferenceText(referenceText);
  const yearStr = info.year ? `(${info.year}).` : '(n.d.).';
  const doiStr = info.doi ? ` https://doi.org/${info.doi}` : '';
  return `${info.authors} ${yearStr} ${info.cleanTitle}.${doiStr}`;
}

/**
 * Formats reference info into IEEE style.
 */
export function generateIEEE(referenceText: string, index: number = 1): string {
  const info = parseReferenceText(referenceText);
  const yearStr = info.year ? `, ${info.year}` : '';
  const doiStr = info.doi ? `, doi: ${info.doi}.` : '.';
  return `[${index}] ${info.authors}, "${info.cleanTitle}"${yearStr}${doiStr}`;
}

/**
 * Exports multiple reference texts into a downloadable file (BibTeX, RIS, APA, IEEE).
 */
export function exportAllReferencesFormat(
  references: string[],
  format: 'bibtex' | 'ris' | 'apa' | 'ieee' = 'bibtex',
  filename: string = "references"
): void {
  let content = "";
  let mimeType = "text/plain;charset=utf-8";
  let extension = "txt";

  if (format === 'bibtex') {
    content = references.map((ref, idx) => generateBibTeX(ref, idx + 1)).join("\n\n");
    extension = "bib";
  } else if (format === 'ris') {
    content = references.map((ref) => generateRIS(ref)).join("\n");
    extension = "ris";
  } else if (format === 'apa') {
    content = references.map((ref) => generateAPA(ref)).join("\n\n");
    extension = "txt";
  } else if (format === 'ieee') {
    content = references.map((ref, idx) => generateIEEE(ref, idx + 1)).join("\n");
    extension = "txt";
  }

  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filename}.${extension}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Alias export for backward compatibility with BibTeX button
 */
export function exportAllBibTeX(references: string[], filename: string = "references.bib"): void {
  exportAllReferencesFormat(references, 'bibtex', filename.replace(/\.bib$/, ''));
}
