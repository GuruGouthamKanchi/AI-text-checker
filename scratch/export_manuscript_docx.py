import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

md_path = r"C:\Users\Goutham\.gemini\antigravity-ide\brain\494f7a94-1825-42e2-95a6-de48a89abf9a\ieee_research_paper_full_manuscript.md"
docx_path = r"d:\AI_Text_Checker\IEEE_Research_Paper_Manuscript.docx"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

doc = docx.Document()

# Set standard margins
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# Helper function to process markdown lines
lines = text.split("\n")
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if not line:
        i += 1
        continue
        
    if line.startswith("# "):
        # Main Title
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line[2:].strip())
        run.font.size = Pt(20)
        run.bold = True
        run.font.color.rgb = RGBColor(0, 51, 102)
    elif line.startswith("## SECTION") or line.startswith("## Abstract") or line.startswith("## REFERENCES"):
        # Major Heading
        p = doc.add_paragraph()
        run = p.add_run(line[3:].strip())
        run.font.size = Pt(14)
        run.bold = True
        run.font.color.rgb = RGBColor(0, 51, 102)
    elif line.startswith("### "):
        # Subheading
        p = doc.add_paragraph()
        run = p.add_run(line[4:].strip())
        run.font.size = Pt(12)
        run.bold = True
        run.font.color.rgb = RGBColor(51, 51, 51)
    elif line.startswith("```"):
        # Code / Diagram / Text Block
        code_block = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            code_block.append(lines[i])
            i += 1
        p = doc.add_paragraph()
        p_text = "\n".join(code_block)
        run = p.add_run(p_text)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(30, 30, 30)
    elif line.startswith("|") and "|" in line[1:]:
        # Table Parsing
        table_data = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            row = [cell.strip() for cell in lines[i].strip().split("|")[1:-1]]
            if row and not all(c.startswith("-") or c == "" for c in row):
                table_data.append(row)
            i += 1
        i -= 1 # adjust step
        if table_data:
            t = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
            t.style = 'Table Grid'
            for r_idx, row in enumerate(table_data):
                for c_idx, val in enumerate(row):
                    clean_val = re.sub(r'\*\*(.*?)\*\*', r'\1', val)
                    clean_val = clean_val.replace("$", "").replace("\\text{", "").replace("}", "")
                    cell = t.cell(r_idx, c_idx)
                    cell.text = clean_val
                    if r_idx == 0:
                        for p_in in cell.paragraphs:
                            for r_in in p_in.runs:
                                r_in.bold = True
            doc.add_paragraph() # spacing
    else:
        # Standard Paragraph
        p = doc.add_paragraph()
        # Clean inline markdown bold / italic
        clean_text = line.replace("**", "").replace("*", "")
        clean_text = clean_text.replace("$$", "").replace("$", "")
        p.add_run(clean_text)
    i += 1

doc.save(docx_path)
print(f"Successfully generated full IEEE manuscript docx: {docx_path}")
