from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

base = Path(__file__).resolve().parent.parent
src = base / 'output/nuclear_morphology_presentation/Nuclear_morphology_study_plan_v13_white.pptx'
dst = base / 'build_ppt/font_normalized_candidate.pptx'
a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
ns = {'a': a}
with ZipFile(src) as source, ZipFile(dst, 'w', ZIP_DEFLATED) as target:
    for item in source.infolist():
        data = source.read(item.filename)
        if item.filename.startswith('ppt/') and item.filename.endswith('.xml'):
            root = E.fromstring(data)
            for el in root.xpath('//a:defRPr | //a:rPr | //a:endParaRPr', namespaces=ns):
                el.set('b', '0')
                for tag in ('latin', 'ea', 'cs'):
                    font = el.find(f'{{{a}}}{tag}')
                    if font is None:
                        font = E.SubElement(el, f'{{{a}}}{tag}')
                    font.set('typeface', 'Arial')
            for el in root.xpath('//a:majorFont/a:latin | //a:minorFont/a:latin', namespaces=ns):
                el.set('typeface', 'Arial')
            data = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        target.writestr(item, data)
print(dst)
