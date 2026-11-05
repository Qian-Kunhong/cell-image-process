from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'reports'
OUT.mkdir(exist_ok=True)
INK='#17324D'; TEAL='#087F83'; GRAY='#657482'
FONT='C:/Windows/Fonts/arial.ttf'
def label(draw,xy,text,size=40,color=INK):
    draw.multiline_text(xy,text,font=ImageFont.truetype(FONT,size),fill=color,anchor='mm',align='center',spacing=10)
def box(draw,xy,text,teal=False):
    draw.rounded_rectangle(xy,radius=18,fill='#EAF6F3' if teal else '#EDF4F8',outline=TEAL if teal else INK,width=4)
    label(draw,((xy[0]+xy[2])/2,(xy[1]+xy[3])/2),text,40,TEAL if teal else INK)
def arrow(draw,a,b,dashed=False):
    color=TEAL if dashed else GRAY
    dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
    if dashed:
        for t in range(0,int(length)-18,32):draw.line((a[0]+ux*t,a[1]+uy*t,a[0]+ux*min(t+18,length-18),a[1]+uy*min(t+18,length-18)),fill=color,width=5)
    else:draw.line((a,b),fill=color,width=5)
    draw.polygon([b,(b[0]-ux*24-uy*11,b[1]-uy*24+ux*11),(b[0]-ux*24+uy*11,b[1]-uy*24-ux*11)],fill=color)
img=Image.new('RGB',(2100,630),'white');draw=ImageDraw.Draw(img)
box(draw,(650,20,1450,180),'Latent regulatory state')
box(draw,(20,365,790,575),'Early nuclear phenotype\nShape, texture, heterogeneity')
box(draw,(1310,365,2080,575),'Final differentiation quality\nPopulation outcome',True)
arrow(draw,(790,185),(405,357));arrow(draw,(1310,185),(1695,357))
arrow(draw,(800,470),(1300,470),True);label(draw,(1050,403),'Prediction',36,TEAL)
img.save(ROOT/'hypothesis.png',dpi=(300,300))
img=Image.new('RGB',(2100,360),'white');draw=ImageDraw.Draw(img)
labels=['Quantify\nnuclear features','Represent\npopulations','Predict\nfuture outcome','Interpret and\nvalidate']
questions=['Stable signal?','Added information?','Beyond density?','Biological consistency?']
for i in range(4):
    x=10+i*535;box(draw,(x,25,x+470,235),labels[i],i==2)
    label(draw,(x+235,300),questions[i],32,GRAY)
    if i<3:arrow(draw,(x+475,130),(x+525,130))
img.save(ROOT/'workflow.png',dpi=(300,300))

d=Document();sec=d.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Inches(.67);sec.bottom_margin=Inches(.67)
sec.left_margin=Inches(.78);sec.right_margin=Inches(.78)
sec.footer_distance=Inches(.3)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    st=d.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
    st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    for a in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
        st.element.get_or_add_rPr().rFonts.attrib.pop(qn('w:'+a),None)
    for border in list(st.element.xpath('./w:pPr/w:pBdr')):
        border.getparent().remove(border)
    grid=OxmlElement('w:snapToGrid');grid.set(qn('w:val'),'0');st.element.get_or_add_pPr().append(grid)
normal=d.styles['Normal'];normal.font.size=Pt(11)
normal.paragraph_format.line_spacing=Pt(15.5);normal.paragraph_format.space_after=Pt(6)
for name,size in [('Title',20),('Subtitle',11),('Heading 1',13),('Heading 2',11),('Caption',9)]:
    st=d.styles[name];st.font.size=Pt(size)
    st.paragraph_format.line_spacing=1.0
    st.paragraph_format.space_before=Pt(9 if 'Heading' in name else 0)
    st.paragraph_format.space_after=Pt(5)
    if 'Heading' in name:st.font.bold=True
d.styles['Title'].paragraph_format.space_after=Pt(6)
d.styles['Caption'].font.italic=False
d.styles['Caption'].font.bold=False
d.core_properties.title='基于早期核图像表型的干细胞分化品质预测'
d.core_properties.subject='中英双语研究计划与计算框架示意图'
d.core_properties.author=''
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');foot._p.append(f)

def p(t,boldlead=None):
    q=d.add_paragraph()
    if boldlead and t.startswith(boldlead):q.add_run(boldlead).bold=True;q.add_run(t[len(boldlead):])
    else:q.add_run(t)
    return q
def h(t):d.add_heading(t,level=1)
def pic(file,caption):
    q=d.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    q.paragraph_format.line_spacing=1.0
    q.paragraph_format.space_after=Pt(2);q.paragraph_format.keep_with_next=True
    shape=q.add_run().add_picture(str(ROOT/file),width=Inches(6.82))
    shape._inline.docPr.set('descr',caption)
    d.add_paragraph(caption,'Caption')
def item(title,body):
    q=p(title+' '+body,boldlead=title)
    q.paragraph_format.keep_together=True
def refs():
    q=d.add_paragraph()
    q.paragraph_format.space_before=Pt(5);q.paragraph_format.space_after=Pt(0)
    r=q.add_run('Selected references / 主要参考文献\n');r.bold=True;r.font.size=Pt(9)
    entries=[
    '[1] A live-cell image-based machine learning strategy for reducing variability in PSC differentiation systems (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10244346/',
    '[2] Image-based discrimination of the early stages of mesenchymal stem cell differentiation (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11321037/',
    '[3] High-content image informatics of the structural nuclear protein NuMA parses trajectories for stem/progenitor cell lineages and oncogenic transformation (2017). https://pubmed.ncbi.nlm.nih.gov/28034673/'
    ]
    for text in entries:
        q=d.add_paragraph();q.paragraph_format.space_after=Pt(3);q.paragraph_format.line_spacing=1.0
        r=q.add_run(text);r.font.size=Pt(8)

d.add_paragraph('早期核表型预测干细胞分化品质','Title')
d.add_paragraph('研究计划概要  中文版','Subtitle')
p('本研究拟检验早期核图像能否提供可解释的预测信息，在最终分化结果出现之前评估培养物品质。研究以群体分化效率为主要目标，以OCT4、YAP等标记读数辅助解释预测依据。')
h('一 背景与知识缺口')
p('干细胞品质评价常依赖后期分子标记、图像定量或功能终点。这些方法能确认培养结果，但反馈较晚，部分评价还会终止样本的后续使用，因而难以支持早期质量判断。已有研究使用明场图像预测PSC分化表现[1]，也利用核形态和核内组织特征区分分化阶段或谱系相关状态[2,3]。')
p('尚待明确的是：在目标分化体系中，早期核表型能否提供超越细胞密度和简单形态的预测信息，以及单细胞状态的异质性如何对应群体最终品质。本课题将这一问题限定为核表型的增量预测价值及其生物学可解释性。')
h('二 生物学假设')
p('染色质组织、核结构及其他内部调控状态，可能同时影响可见的核表型和细胞后续的分化响应。因此，核形态与纹理可以作为潜在状态的图像代理；群体内这些特征的分布及空间组织，则可能反映不同分化潜能细胞的组成。核心假设是：早期核表型及其群体异质性包含可预测后期分化效率的信息。')
pic('hypothesis.png','图1  假说关系图。实线表示假定的状态影响，虚线表示待验证的预测关系；图像关联本身不证明因果机制。')
h('三 Objective')
p('建立可解释的计算表型框架，以早期核图像直接预测培养孔或克隆的最终分化效率。主要标签为预先定义的目标细胞比例或分化效率；OCT4、YAP及其他终点图像读数用于解释模型，不自动等同于最终品质。')
p('活细胞核图像的非终止性及低扰动性是应用前提。现有DAPI图像用于表型发现，向Hoechst活细胞图像迁移时需验证特征与模型的适用性。')
d.add_page_break()
h('四 预期研究项目')
pic('workflow.png','图2  计算验证流程。各步骤依次检验可测性、群体信息增益、未来预测能力与解释的一致性。')
item('1 核表型定量与稳定性评价','通过核分割、图像质量控制和特征提取，建立核形态、纹理与邻域信息的统一表示。评估分割误差、亮度变化和批次差异对表型的影响，回答“图像信号是否稳定可测”，为后续预测提供可信输入。')
item('2 单细胞到群体的多尺度表示','将单细胞特征汇总为分布宽度、表型比例和空间组织指标，并与仅使用均值的表示比较。通过特征消融检验异质性与空间信息是否带来增益，回答“群体品质需要哪些层级的信息”。')
item('3 早期表型对未来分化效率的预测','按培养孔或克隆关联早期图像与后期真实结局，在预设时间窗建立回归模型。与密度、细胞数量和核面积等简单基线比较，评价误差及最早有效预测时间，回答“核表型能否提前预测，而非仅识别同时期状态”。')
item('4 生物学解释与独立验证','将预测性表型与OCT4、YAP等终点读数进行关联，检验解释是否符合生物学预期。同时按独立培养单位或批次划分训练与测试数据，评价泛化、校准和不确定性。该步骤回答“预测依据是否可信且可重复”；特征归因作为模型解释，不作为机制证明。')
h('五 总结')
p('若假说成立，本研究将建立早期核表型与群体后期分化品质之间的可验证联系，补充现有核表型描述与最终质量预测之间的证据。其应用价值在于提前识别低品质风险，并评估减少等待时间和终点评价成本的可能性。预测增益、提前时间及不确定性将共同决定该框架是否值得应用。')
refs()
d.add_page_break()
d.add_paragraph('Early Nuclear Image Phenotyping for\nPrediction of Stem Cell Differentiation Quality','Title')
d.add_paragraph('Research proposal summary  English version','Subtitle')
h('1 Background and knowledge gap')
p('Stem cell quality assessment often relies on late molecular, imaging or functional endpoints. These measurements confirm culture outcomes but provide delayed feedback, and some prevent further use of the assessed sample. Previous studies have predicted PSC differentiation performance from bright-field images [1] and distinguished differentiation stages or lineage-associated states using nuclear features [2,3].')
p('For the target differentiation system, it remains to be established whether early nuclear phenotypes add predictive information beyond cell density and simple morphology, and how cellular heterogeneity relates to final population quality. This study addresses that specific gap in predictive value and interpretability.')
h('2 Biological hypothesis')
p('Chromatin organization, nuclear architecture and other regulatory states may influence both observable nuclear phenotypes and subsequent differentiation responses. Nuclear shape and texture may therefore serve as image-derived proxies for latent state. Their distributions and spatial organization may further capture the composition of cells with different differentiation competence. We hypothesize that early nuclear phenotypes and population heterogeneity contain information predictive of later differentiation efficiency.')
pic('hypothesis.png','Figure 1. Solid arrows indicate hypothesized state effects. The dashed arrow represents a predictive association to be tested, rather than evidence of a causal mechanism.')
h('3 Objective')
p('Develop an interpretable computational framework that predicts final differentiation efficiency for a well or colony from early nuclear images. The primary target is a predefined differentiation efficiency or target-cell proportion. OCT4, YAP and other endpoint measurements provide explanatory indicators and are not assumed to establish final quality.')
p('Application assumes that live-cell nuclear imaging is non-terminal and minimally perturbative. Existing DAPI data support phenotype discovery, while transfer to live-cell Hoechst images requires validation of feature and model compatibility.')
d.add_page_break()
h('4 Expected research projects')
pic('workflow.png','Figure 2. Each stage tests a necessary link between measurable nuclear phenotypes and interpretable prediction of future population quality.')
item('1 Quantify nuclear phenotypes and assess stability.','Combine nuclear segmentation, image quality control and feature extraction to represent shape, texture and neighborhood information. Test sensitivity to segmentation errors, image intensity and batch variation. This establishes whether the proposed signal is sufficiently stable to support prediction.')
item('2 Construct population representations.','Aggregate single-cell measurements into feature distributions, phenotype proportions and spatial descriptors. Compare these representations with mean-feature models and use ablation to measure the contribution of each information source. This tests whether heterogeneity and spatial organization improve population quality assessment.')
item('3 Predict future differentiation efficiency.','Link early images to later outcomes at the well or colony level and fit regression models within predefined early time windows. Compare performance with cell density, cell count and nuclear-area baselines. Prediction error and the earliest informative time point will establish whether nuclear features forecast future outcomes rather than merely recognize current states.')
item('4 Interpret predictions and test generalization.','Associate predictive phenotypes with OCT4, YAP and other endpoint measurements to assess biological consistency. Separate training and testing by independent culture unit or batch, and evaluate generalization, calibration and uncertainty. Feature attribution will explain model behavior without being treated as proof of a biological mechanism.')
h('5 Summary')
p('If supported, the hypothesis will establish a testable relationship between early nuclear phenotypes and later population differentiation quality. The framework will address delayed endpoint feedback by identifying quality risks earlier. Its practical value will depend on the gain over simple baselines, prediction lead time and uncertainty, with potential reductions in waiting time and endpoint assessment costs.')
refs()
for i, paragraph in enumerate(list(d.paragraphs)):
    if paragraph._p.xpath('.//w:br[@w:type="page"]'):
        following = paragraph._p.getnext()
        if following is not None:
            from docx.text.paragraph import Paragraph
            Paragraph(following, d._body).paragraph_format.page_break_before = True
        paragraph._p.getparent().remove(paragraph._p)
target=OUT/'Stem_Cell_Research_Plan_Report.docx'
d.save(target)
print(target)
