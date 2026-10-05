import fitz

pdf_path = 'd:/AI_Text_Checker/LLM_Architectures_and_Training_Survey.pdf'

# Test 1: page.draw_rect
doc1 = fitz.open(pdf_path)
page1 = doc1[0]
rect1 = fitz.Rect(50, 50, 400, 200)
page1.draw_rect(rect1, color=None, fill=(186/255, 45/255, 34/255), fill_opacity=0.3)
pix1 = page1.get_pixmap(dpi=150)
pix1.save('scratch/test_draw_rect.png')
doc1.close()

# Test 2: Shape
doc2 = fitz.open(pdf_path)
page2 = doc2[0]
shape = page2.new_shape()
shape.draw_rect(rect1)
shape.finish(color=None, fill=(186/255, 45/255, 34/255), fill_opacity=0.3)
shape.commit()
pix2 = page2.get_pixmap(dpi=150)
pix2.save('scratch/test_shape.png')
doc2.close()

# Test 3: Highlight annotation
doc3 = fitz.open(pdf_path)
page3 = doc3[0]
annot = page3.add_highlight_annot(rect1)
annot.set_colors(stroke=(1, 0, 0))
annot.update()
pix3 = page3.get_pixmap(dpi=150)
pix3.save('scratch/test_annot.png')
doc3.close()

print("Saved 3 test images in scratch/")
