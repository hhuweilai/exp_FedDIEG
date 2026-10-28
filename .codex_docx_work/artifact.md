# DOCX Template Distillation: Linux Camera Interview Materials

Reference: `F:\秋招\面试知识库\基于Linux的摄像头模组缺陷检测系统面试资料.docx`

- SHA-256: `75b5086b586f86d85f302bc836f9cae52b62ec0fe3a923570ca3c9bd23ff0be3`
- Size: 40912 bytes
- Sections: 1
- Page count: unresolved because `render_docx.py` failed due missing `soffice` executable.

Page System

- Page size: A4 portrait, 8.27 x 11.69 in.
- Margins: left 1.25 in, right 1.25 in, top 1.00 in, bottom 1.00 in.
- Header/footer: linked to previous, no different first page observed.

Typography And Paragraph Roles

- Main font evidence: Chinese body text uses Songti/宋体, 10.5 pt in key body paragraphs.
- Title role: `Heading 1`, used for project name and later major numbered sections.
- Section role: `Heading 2`, used for "项目核心词汇", "项目开发板基本信息", "项目介绍", "项目工作流程", and "核心技术原理".
- Body role: mostly `Normal` or `Normal (Web)`. Some opening paragraphs use left indent around 22 pt.
- Subtopic role: short unnumbered `Normal` paragraphs or `List Paragraph` paragraphs, often followed by explanatory `Normal` paragraphs.
- Bullet role: `List Bullet`, with left indent about 27 pt and hanging indent about -13.55 pt.

Tables

- All observed tables use `Normal Table`.
- Table patterns:
  1. Two-column glossary table: `分类 | 核心词汇`.
  2. Two-column basic information table: `项目 | 信息`.
  3. Four-column protocol/event table.
  4. One-cell note/table block for compact procedural explanation.
  5. Three-column workflow/thread table.
  6. Four-column file/module explanation table.
  7. Two-column interview Q&A table: `面试问题 | 回答模板`.
- Tables are content tables, not decorative layout tables. Use explicit widths in the generated document to avoid default cramped cells.

Content Flow To Preserve

1. Project name as the first `Heading 1`.
2. "项目核心词汇" heading followed by a glossary table.
3. "项目基本信息" heading followed by a key-value table.
4. "项目介绍" heading with concise project overview paragraphs.
5. "项目工作流程" heading with staged workflow paragraphs.
6. "核心技术原理" heading with subtopics and interview-style explanations.
7. "关键模块逐文件说明" major heading with a file/module mapping table.
8. "面试常见问题与回答模板" major heading with a Q&A table.
9. "容易被追问的细节" major heading with short bullet points.

Editable Slot Map

- Replace all domain-specific Linux/camera content with FedDIEG content.
- Preserve the overall order, heading types, table-first overview style, and direct interview-answer tone.
- Use FedDIEG repository facts for datasets, scripts, modules, baselines, metrics, and limitations.
- Output path must be different from the reference file.

Fidelity Gates

- Reference file must remain unchanged.
- Generated document should use A4, 1.25 in side margins, Songti/宋体 body text, and simple Word heading/table structure.
- Render gate attempted through `render_docx.py`; if `soffice` remains unavailable, structural QA and package checks are used instead.
