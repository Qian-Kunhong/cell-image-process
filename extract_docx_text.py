from zipfile import ZipFile
from pathlib import Path
import xml.etree.ElementTree as ET
path = Path(r'C:\Users\dodos\Downloads\Immunostaining Training Report_Ekindone_26.11.docx')
out = Path('docx_extracted_text.txt')
with ZipFile(path) as z:
    xml = z.read('word/document.xml')
root = ET.fromstring(xml)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paras = []
for p in root.findall('.//w:p', ns):
    line = ''.join((t.text or '') for t in p.findall('.//w:t', ns)).strip()
    if line:
        paras.append(line)
out.write_text('\n'.join(paras), encoding='utf-8')
print(f'wrote {out} paras={len(paras)}')
