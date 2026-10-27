import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/dodos/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const workspaceDir = path.resolve(".");
const buildDir = path.join(workspaceDir, "build_ppt");
const assetDir = path.join(buildDir, "assets");
const previewDir = path.join(buildDir, "previews_v13");
const outputDir = path.join(workspaceDir, "output", "nuclear_morphology_presentation");
const FINAL_PPTX = path.join(outputDir, "Nuclear_morphology_study_plan_v13_white.pptx");
const SKILL_DIR = "C:\\Users\\dodos\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.1013.11914\\skills\\presentations";
const RUNTIME_PYTHON = "C:\\Users\\dodos\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
process.env.RUNTIME_NODE_MODULES = "C:\\Users\\dodos\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { navy:"#FFFFFF", navy2:"#F3F4F6", blue:"#111111", cyan:"#111111", pale:"#111111", white:"#111111", muted:"#5F6368", line:"#D1D5DB", warm:"#111111", red:"#111111", green:"#111111" };
const FONT = "Aptos";

function rect(slide,x,y,w,h,fill,r="rounded-md",line="none") { return slide.shapes.add({ geometry:r==="none"?"rect":"roundRect", position:{left:x,top:y,width:w,height:h}, fill, line:line==="none"?{style:"solid",fill:"none",width:0}:{style:"solid",fill:line,width:1}, ...(r==="none"?{}:{borderRadius:r}) }); }
function textBox(slide,text,x,y,w,h,o={}) { const q=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}}); q.text=text; q.text.style={fontFamily:FONT,fontSize:o.size??22,bold:o.bold??false,color:o.color??C.white,alignment:o.align??"left",verticalAlignment:o.valign??"top"}; return q; }
function title(slide,h,sub="") { textBox(slide,h,70,42,1120,52,{size:32,bold:true}); rect(slide,70,108,68,5,C.cyan,"none"); if(sub) textBox(slide,sub,158,98,1030,30,{size:16,color:C.muted}); }
function footer(slide,n,section="") { if(section) textBox(slide,section.toUpperCase(),70,669,180,20,{size:11,bold:true,color:C.muted}); if(section==="Published study") textBox(slide,"Kim et al., Biotechnol. Bioeng. 2023;120:593-607. doi:10.1002/bit.28285",260,668,820,20,{size:10,color:C.muted,align:"center"}); textBox(slide,String(n).padStart(2,"0"),1170,668,44,20,{size:11,color:C.muted,align:"right"}); }
async function bytes(name){const b=await fs.readFile(path.join(assetDir,name));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}
async function addImage(slide,name,x,y,w,h,fit="cover") { return slide.images.add({blob:await bytes(name),contentType:"image/png",alt:name,fit,position:{left:x,top:y,width:w,height:h}}); }
function note(slide,t){slide.speakerNotes.text=t;}
function bullet(slide,head,body,x,y,w,color=C.cyan){ textBox(slide,head,x,y,w,30,{size:20,bold:true,color}); textBox(slide,body,x,y+34,w,58,{size:18,color:C.pale}); }

// 1 Cover
{
 const s=deck.slides.add(); s.background.fill=C.navy; await addImage(s,"D21_HA10_DAPI.png",735,0,545,720,"cover"); rect(s,705,0,70,720,C.navy2,"none");
 textBox(s,"RESEARCH PROPOSAL",74,78,380,26,{size:15,bold:true,color:C.cyan});
 textBox(s,"Early prediction of cell quality using nuclear morphology",74,145,590,210,{size:46,bold:true});
 textBox(s,"Experimental plan based on published hepatocyte differentiation data",76,382,590,70,{size:23,color:C.pale});
 textBox(s,"From early images to later quality outcomes",76,555,500,30,{size:17,color:C.muted});
 note(s,"Cover image source: 10× DAPI image, D21 HA10, from the supplied hepatocyte differentiation dataset. Display contrast normalized consistently across the 16-image series.");
}

// 2 Paper story
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Paper story","Temporary control of cell-cell adhesion before differentiation was tested as a way to reduce outcome variability"); footer(s,2,"Published study");
 textBox(s,"Problem",82,150,220,32,{size:18,bold:true,color:C.red,align:"center"});
 textBox(s,"hiPSC cultures develop spatial and mechanical heterogeneity during expansion",82,208,220,112,{size:22,bold:true,align:"center",valign:"middle"});
 textBox(s,"HA treatment",382,150,220,32,{size:18,bold:true,color:C.cyan,align:"center"});
 textBox(s,"Botulinum hemagglutinin temporarily disrupts E-cadherin-mediated adhesion",382,208,220,112,{size:22,bold:true,align:"center",valign:"middle"});
 textBox(s,"Measured response",682,150,220,32,{size:18,bold:true,color:C.blue,align:"center"});
 textBox(s,"F-actin and YAP distributions change before differentiation",682,208,220,112,{size:22,bold:true,align:"center",valign:"middle"});
 textBox(s,"Outcome",982,150,220,32,{size:18,bold:true,color:C.warm,align:"center"});
 textBox(s,"Hepatocyte differentiation becomes more efficient and more consistent for key endpoints",982,208,220,112,{size:22,bold:true,align:"center",valign:"middle"});
 for(const x of [302,602,902]){rect(s,x,258,80,4,C.line,"none");textBox(s,">",x+36,238,40,42,{size:28,bold:true,color:C.muted,align:"center"});}
 rect(s,80,390,1120,2,C.line,"none");
 textBox(s,"The paper's central claim",90,420,300,30,{size:18,bold:true,color:C.cyan});
 textBox(s,"Repeated HA exposure synchronizes behavior-driven mechanical memory before differentiation, which improves the efficiency and reproducibility of hepatocyte differentiation.",90,466,1080,92,{size:29,bold:true,align:"center",valign:"middle"});
 textBox(s,"Evidence supports the sequence of associations. The proposed causal mechanism is the authors' interpretation.",90,598,1080,34,{size:17,color:C.muted,align:"center"});
 note(s,"Source: Kim MH, Thanuthanakhun N, Kino-oka M. Biotechnology and Bioengineering. 2023;120:593-607. DOI: 10.1002/bit.28285. This slide separates observed measurements from the authors' mechanistic interpretation.");
}

// 3 Published experimental conditions and data
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"\u00A0Published experimental conditions","The paper used two related comparisons during the same five-day hiPSC expansion period"); footer(s,3,"Published study");
 textBox(s,"COMMON CELL CULTURE",72,145,300,26,{size:15,bold:true,color:C.muted});
 textBox(s,"1383D2 hiPSCs",72,181,250,30,{size:22,bold:true,color:C.cyan});
 textBox(s,"Seeded at 2.5 x 10³ cells/cm² on iMatrix-511 (0.25 microgram/cm²). Cultured for 5 days in StemFit AK02N. Y-27632 was used at 10 micromolar during seeding.",72,220,380,105,{size:18,color:C.pale});
 textBox(s,"HA COMPARISONS",72,345,300,26,{size:15,bold:true,color:C.muted});
 textBox(s,"Mechanism experiment",72,382,260,28,{size:20,bold:true,color:C.blue});
 textBox(s,"No HA; one 10 nM exposure for 24 h from Day -3; or 10 nM once daily from Day -3 to Day 0.",72,422,380,66,{size:17,color:C.pale});
 textBox(s,"Dose and outcome experiment",72,505,310,28,{size:20,bold:true,color:C.warm});
 textBox(s,"0, 1, 5 or 10 nM HA once daily from Day -3 to Day 0.",72,545,380,52,{size:17,color:C.pale});
 await addImage(s,"Figure_1.png",490,145,725,430,"contain");
 rect(s,72,615,1140,2,C.line,"none");
 textBox(s,"Day 0",82,631,90,26,{size:18,bold:true,color:C.cyan});
 textBox(s,"Hepatocyte differentiation started directly without passaging; medium was changed daily.",175,627,1000,34,{size:19,bold:true,color:C.pale});
 note(s,"Source: article Methods 2.1 and 2.3 and published Figure 1. Differentiation started at Day 0 without passaging. Media were changed daily. The Methods specify 7 days for the third stage, while outcomes were also measured at Day 21; the slide therefore states D10 onward rather than assigning an unsupported end day.");
}

// 4 Published measurements
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"\u00A0Published measurements and sample sizes","Each time point contributes a different part of the paper's evidence"); footer(s,4,"Published study");
 const rows=[
  ["D0","Before differentiation","Phase contrast; OCT3/4-SSEA4 flow cytometry; E-cadherin/F-actin and YAP/DAPI imaging; western blots","Flow: 10 wells; YAP: 206-355 cells per condition"],
  ["D3","Definitive endoderm","SOX2/SOX17 immunostaining and SOX17 qRT-PCR","qRT-PCR: 3 independent experiments"],
  ["D17","Hepatocyte differentiation","Marker imaging; ALB/AFP flow cytometry; albumin and urea secretion","Within-run: 10 wells; between-run: 3 runs"],
  ["D21","Final reported outcome","Marker imaging and qRT-PCR; ALB/AFP flow cytometry; albumin and urea secretion","Within-run: 10 wells; between-run: 3 runs"]
 ];
 textBox(s,"TIME",72,150,90,28,{size:15,bold:true,color:C.muted});
 textBox(s,"STAGE",180,150,210,28,{size:15,bold:true,color:C.muted});
 textBox(s,"DATA REPORTED",420,150,500,28,{size:15,bold:true,color:C.muted});
 textBox(s,"REPORTED n",955,150,245,28,{size:15,bold:true,color:C.muted});
 rows.forEach((r,i)=>{const y=188+i*101;rect(s,70,y,1130,86,i%2===0?C.navy2:"#FAFAFA","rounded-md",i===0?C.cyan:C.line);textBox(s,r[0],88,y+24,72,36,{size:23,bold:true,color:C.white,align:"center"});textBox(s,r[1],180,y+13,210,58,{size:18,bold:true,color:C.pale,valign:"middle"});textBox(s,r[2],420,y+9,500,66,{size:17,color:C.pale,valign:"middle"});textBox(s,r[3],955,y+10,225,64,{size:17,bold:true,color:C.muted,valign:"middle"});});
 rect(s,72,608,1128,2,C.line,"none");
 textBox(s,"Differentiation protocol",80,625,260,26,{size:16,bold:true,color:C.cyan});
 textBox(s,"D0-D3: RPMI/B27 + activin A 100 ng/mL + Wnt3a 50 ng/mL; D3-D10: KO-DMEM + 20% KSR + 1% DMSO; D10 onward: HCM + HGF 20 ng/mL + oncostatin M 20 ng/mL.",345,614,840,52,{size:16,bold:true,color:C.pale});
 note(s,"Sources: article Methods 2.4-2.11 and figure captions for Figures 1, 3, 4 and 7. Folder formats were verified in the supplied raw-data directory. Figure 1 reports n=10 for OCT3/4-SSEA4 and cell density; Figure 3 reports 206-355 cells per condition for YAP N/C; Figure 4 qRT-PCR reports three independent experiments; Figure 7 reports 10 wells for intra-run and three runs for inter-run analysis. These n values do not establish the replicate structure of every individual image file.");
}

// 5 Paper evidence
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Published data before differentiation","Day 0 measurements connect HA exposure to cell state before differentiation"); footer(s,5,"Published study");
 await addImage(s,"Figure_3.png",55,145,690,520,"contain");
 bullet(s,"Images","Phase contrast; E-cadherin and F-actin confocal images; YAP and DAPI confocal images.",785,150,405);
 bullet(s,"Quantitative data","OCT3/4-SSEA4 flow cytometry used 10 wells. YAP nuclear-to-cytoplasmic analysis used 206-355 cells per condition.",785,285,405,C.warm);
 bullet(s,"Protein data","Western blots measured E-cadherin, integrin beta 1, total and phosphorylated YAP, and total and phosphorylated MLC.",785,430,405,C.green);
 textBox(s,"Main result: repeated 10 nM HA produced a narrower YAP distribution and shifted YAP toward the cytoplasm. The authors call this synchronized mechanical memory.",785,560,405,90,{size:18,bold:true,color:C.cyan});
 note(s,"Source: Kim et al., Biotechnology and Bioengineering 120 (2023) 593-607, Results 3.2-3.3 and Figure 3. DOI 10.1002/bit.28285. The causal mechanism in Figure 3f is explicitly described by the authors as a hypothesis.");
}

// 6 Paper outcomes
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Differentiation data and outcomes","D3, D17 and D21 data link the pre-treatment to differentiation quality"); footer(s,6,"Published study");
 bullet(s,"Differentiation efficiency","At Day 21, ALB/AFP double-positive cells increased from 11.5% in control to 78.2% with HA, a 6.79-fold difference.",72,145,430);
 bullet(s,"Functional readouts","At Day 21, albumin secretion was 1.55-fold higher and urea production was 1.53-fold higher with HA.",72,285,430,C.green);
 bullet(s,"Reproducibility","For ALB/AFP double-positive cells, Day 21 CV decreased from 17.18% to 6.19% within one run and from 32.61% to 0.79% across runs.",72,415,430,C.warm);
 textBox(s,"Important limitation",72,552,190,28,{size:17,bold:true,color:C.red});
 textBox(s,"HA reduced total cell density. Day 21 inter-run urea CV increased from 2.24% to 24.62%, so reproducibility did not improve for every endpoint.",72,583,430,68,{size:18,bold:true,color:C.pale});
 await addImage(s,"Figure_5.png",530,135,685,520,"contain");
 note(s,"Source: Kim et al., Biotechnology and Bioengineering 120 (2023) 593-607, Results 3.4, published Figure 7 and Table 1. The supplied submission Figure 5.tif corresponds to the final differentiation outcome figure. Intra-run analysis used 10 wells from one run; inter-run analysis used three independent runs. Table 1 reports Day 21 inter-run urea CV of 2.24% without HA and 24.62% with HA, which conflicts with the article text stating that all Day 21 inter-run CV values with HA were below 10%.");
}

// 7 Gap
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Research gap","The dataset contains a common image channel that the paper did not use as a unified readout"); footer(s,7,"Research gap");
 const left=["Different markers at different stages","Separate analyses for each endpoint","Limited connection between early images and later outcomes"];
 const right=["DAPI is present across imaging panels","The same feature space can compare all samples","Morphology distributions can support state and heterogeneity analysis"];
 textBox(s,"Published analysis",98,155,430,36,{size:25,bold:true,color:C.warm});
 textBox(s,"Proposed analysis",752,155,430,36,{size:25,bold:true,color:C.cyan});
 left.forEach((v,i)=>{rect(s,92,220+i*112,440,78,C.navy2,"rounded-lg",C.line);textBox(s,v,116,234+i*112,390,50,{size:19,color:C.pale,valign:"middle"});});
 right.forEach((v,i)=>{rect(s,746,220+i*112,440,78,"#F3F4F6","rounded-lg",C.cyan);textBox(s,v,770,234+i*112,390,50,{size:19,color:C.pale,valign:"middle"});});
 textBox(s,"Core question",555,280,170,28,{size:15,bold:true,color:C.blue,align:"center"});
 textBox(s,"Can nuclear morphology describe experimental state and support later prediction?",545,320,190,150,{size:23,bold:true,align:"center",valign:"middle"});
}

// 8 Our objective
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Our study objective"); footer(s,8,"Overall plan");
 textBox(s,"Build a nuclear morphology framework for early prediction of later cell quality.",105,150,1070,82,{size:32,bold:true,align:"center",valign:"middle"});
 const items=[["Cell level","Each nucleus becomes a feature vector"],["Sample level","Each image becomes a morphology distribution"],["Study level","Samples form a shared state space"]];
 items.forEach((it,i)=>{const x=80+i*400;rect(s,x,310,345,188,i===2?"#EDEFF2":C.navy2,"rounded-lg",i===2?C.cyan:C.line);textBox(s,it[0],x+25,340,295,34,{size:23,bold:true,color:C.blue,align:"center"});textBox(s,it[1],x+38,400,269,65,{size:19,color:C.pale,align:"center",valign:"middle"});});
 textBox(s,"Primary outputs",105,552,180,28,{size:17,bold:true,color:C.warm});
 textBox(s,"Early morphology score, predicted cell quality and confidence of the prediction",290,546,880,48,{size:21,bold:true});
}

// 9 Overall plan
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"What the study will test","Central hypothesis: early nuclear morphology contains information about later cell quality"); footer(s,9,"Overall plan");
 textBox(s,"EARLY IMAGE",86,155,250,28,{size:15,bold:true,color:C.cyan,align:"center"});
 textBox(s,"MORPHOLOGY SIGNAL",515,155,250,28,{size:15,bold:true,color:C.blue,align:"center"});
 textBox(s,"LATER QUALITY",944,155,250,28,{size:15,bold:true,color:C.warm,align:"center"});
 rect(s,82,202,260,145,"#F3F4F6","rounded-lg",C.cyan);
 textBox(s,"Nuclei before or early in differentiation",106,230,212,82,{size:24,bold:true,align:"center",valign:"middle"});
 rect(s,510,202,260,145,C.navy2,"rounded-lg",C.blue);
 textBox(s,"Shape, texture and population heterogeneity",534,230,212,82,{size:24,bold:true,align:"center",valign:"middle"});
 rect(s,938,202,260,145,"#F3F4F6","rounded-lg",C.warm);
 textBox(s,"Cell identity, function and reproducibility",962,230,212,82,{size:24,bold:true,align:"center",valign:"middle"});
 rect(s,342,272,168,4,C.line,"none"); textBox(s,">",468,250,42,42,{size:30,bold:true,color:C.cyan,align:"center"});
 rect(s,770,272,168,4,C.line,"none"); textBox(s,">",896,250,42,42,{size:30,bold:true,color:C.warm,align:"center"});
 const tests=[
  ["1. Detect","Does treatment change nuclear morphology at D0?"],
  ["2. Connect","Which early morphology features relate to later quality measurements?"],
  ["3. Predict","Can an early morphology model estimate later quality in held-out samples?"]
 ];
 tests.forEach((v,i)=>{const x=80+i*400; textBox(s,v[0],x,430,340,34,{size:23,bold:true,color:[C.cyan,C.blue,C.warm][i],align:"center"}); textBox(s,v[1],x+18,478,304,82,{size:19,color:C.pale,align:"center",valign:"middle"});});
 textBox(s,"Study endpoint",86,611,180,26,{size:16,bold:true,color:C.muted});
 textBox(s,"An early, image-based estimate of later cell quality, with validation at the sample level.",270,602,900,44,{size:23,bold:true});
}

// 10 Data map
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"How the available data support the study"); footer(s,10,"Overall plan");
 const rows=[
  ["D0 nuclear images","HA00, HA01, HA05 and HA10","Test whether treatment creates an early morphology signal","Experiment 1"],
  ["D3 nuclear images","Early differentiation stage","Test whether the D0 signal persists or changes","Experiment 2"],
  ["D17 and D21 nuclear images","Late differentiation stages","Describe the final morphology state","Experiment 2"],
  ["Paired marker and outcome data","Same treatment and stage groups","Define cell quality and train the early prediction model","Experiments 3 and 4"]
 ];
 textBox(s,"DATA USED",80,147,250,28,{size:15,bold:true,color:C.muted});
 textBox(s,"COMPARISON",365,147,260,28,{size:15,bold:true,color:C.muted});
 textBox(s,"WHY IT IS NEEDED",660,147,360,28,{size:15,bold:true,color:C.muted});
 textBox(s,"ROLE",1050,147,150,28,{size:15,bold:true,color:C.muted,align:"center"});
 rows.forEach((r,i)=>{const y=188+i*98;rect(s,70,y,1130,82,i===0?"#EDEFF2":C.navy2,"rounded-md",i===0?C.cyan:C.line);textBox(s,r[0],92,y+14,245,52,{size:20,bold:true,color:C.white,valign:"middle"});textBox(s,r[1],365,y+12,260,56,{size:18,color:C.pale,valign:"middle"});textBox(s,r[2],660,y+9,350,62,{size:18,color:C.pale,valign:"middle"});textBox(s,r[3],1038,y+13,145,52,{size:17,bold:true,color:C.muted,align:"center",valign:"middle"});});
 textBox(s,"Why start at D0?",82,603,180,28,{size:17,bold:true,color:C.cyan});
 textBox(s,"D0 tests the earliest usable signal before differentiation adds stage-related variation.",270,596,900,42,{size:22,bold:true});
 note(s,"Primary image source folder: F:\\Naruchit data\\1_Data of published manuscripts\\3_Hepatocyte differentiation\\03_Raw data\\Immunostaining\\IN Cell\\D0_3_17_21_E-cad N-cad\\10x. The matrix represents 16 DAPI images: four stages by four treatment levels.");
}

// 11 First experiment design
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Next experiment: D0 treatment comparison","Question: do four treatment levels produce different nuclear morphology distributions?"); footer(s,11,"Next experiment");
 const files=["D0_HA00_DAPI.png","D0_HA01_DAPI.png","D0_HA05_DAPI.png","D0_HA10_DAPI.png"], labels=["HA00","HA01","HA05","HA10"];
 for(let i=0;i<4;i++){const x=72+i*300;await addImage(s,files[i],x,170,260,260,"cover");textBox(s,labels[i],x,444,260,34,{size:22,bold:true,color:i===3?C.cyan:C.white,align:"center"});}
 textBox(s,"Primary contrast",95,525,190,28,{size:17,bold:true,color:C.blue});
 textBox(s,"HA00 versus HA10",300,519,270,38,{size:24,bold:true});
 textBox(s,"Secondary analysis",650,525,200,28,{size:17,bold:true,color:C.warm});
 textBox(s,"Four-level trend and distribution width",860,519,345,52,{size:22,bold:true});
 textBox(s,"D0 isolates treatment level from stage. The analysis remains exploratory until independent image or well replication is confirmed.",95,608,1090,44,{size:18,color:C.muted});
 note(s,"Image sources: D0 HA00, HA01, HA05 and HA10 DAPI images from the primary 10× series. Display contrast normalized consistently across the 16-image series.");
}

// 12 Pipeline
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Experiment 1 analysis workflow"); footer(s,12,"Next experiment");
 const stages=[["Image QC","Focus, saturation, illumination and empty regions"],["Segmentation","Cellpose masks and border-object removal"],["Feature extraction","Shape-only and full feature sets"],["Block summaries","Median, IQR, CV and subtype proportions"],["Group tests","Shift, classification and heterogeneity"]];
 stages.forEach((v,i)=>{const x=50+i*246;rect(s,x,184,204,236,i===0?"#E9EEF3":C.navy2,"rounded-lg",i===0?C.blue:C.line);textBox(s,String(i+1).padStart(2,"0"),x+18,208,50,26,{size:14,bold:true,color:C.cyan});textBox(s,v[0],x+18,254,168,42,{size:22,bold:true});textBox(s,v[1],x+18,318,168,74,{size:17,color:C.pale});if(i<4)textBox(s,">",x+204,282,42,46,{size:30,bold:true,color:C.line,align:"center"});});
 rect(s,76,472,1120,2,C.line,"none");
 textBox(s,"Feature sets",80,500,180,30,{size:17,bold:true,color:C.warm});
 textBox(s,"Shape only",275,496,170,34,{size:22,bold:true,color:C.cyan});
 textBox(s,"Area, circularity, aspect ratio, eccentricity, solidity and boundary irregularity",275,536,355,74,{size:17,color:C.pale});
 textBox(s,"Full set",690,496,170,34,{size:22,bold:true,color:C.blue});
 textBox(s,"Shape plus DAPI intensity, texture, spatial density and distribution features",690,536,420,74,{size:17,color:C.pale});
}

// 13 Analysis and outputs
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Planned analyses and outputs"); footer(s,13,"Next experiment");
 const rows=[["Feature distributions","Which features shift, broaden or create subgroups?","Violin plots and effect sizes"],["PCA and UMAP","Do treatment groups occupy different morphology regions?","State-space map"],["Classification","Can morphology distinguish HA00 from HA10 and all four levels?","Blocked accuracy and confusion matrix"],["Heterogeneity","Does treatment change within-image dispersion?","CV, IQR and state-space spread"]];
 textBox(s,"Analysis",82,154,240,28,{size:16,bold:true,color:C.muted});textBox(s,"Question",350,154,505,28,{size:16,bold:true,color:C.muted});textBox(s,"Output",890,154,285,28,{size:16,bold:true,color:C.muted});
 rows.forEach((r,i)=>{const y=194+i*92;rect(s,72,y,1120,76,i%2===0?C.navy2:"#FAFAFA","rounded-md");textBox(s,r[0],96,y+13,230,44,{size:20,bold:true,color:C.white,valign:"middle"});textBox(s,r[1],350,y+10,505,48,{size:18,color:C.pale,valign:"middle"});textBox(s,r[2],890,y+10,270,48,{size:18,color:C.pale,valign:"middle"});});
 textBox(s,"Statistical unit",92,594,170,28,{size:17,bold:true,color:C.red});
 textBox(s,"Independent image or spatial block. Individual nuclei describe distributions but do not count as independent experiments.",270,588,880,50,{size:20,bold:true});
}

// 14 Expected outcomes
{
 const s=deck.slides.add(); s.background.fill=C.navy; title(s,"Expected outcomes and decision points","Hypotheses, not observed results"); footer(s,14,"Next experiment");
 const out=[["A. Mean state shift","Group centers move in feature space.","Proceed to the stage map using D0, D3, D17 and D21."],["B. Heterogeneity shift","Group centers overlap, but distribution widths differ.","Use morphology dispersion as the primary treatment response."],["C. No robust signal","Apparent separation disappears after blocked validation.","Review image QC, feature stability and experimental replication."]];
 out.forEach((v,i)=>{const y=150+i*138;rect(s,75,y,8,106,[C.cyan,C.warm,C.red][i],"none");textBox(s,v[0],108,y+1,290,34,{size:24,bold:true,color:[C.cyan,C.warm,C.red][i]});textBox(s,v[1],415,y+1,700,36,{size:20,bold:true});textBox(s,v[2],415,y+48,700,48,{size:18,color:C.muted});});
 rect(s,75,584,1120,2,C.line,"none");
 textBox(s,"Immediate deliverable",82,610,220,28,{size:16,bold:true,color:C.blue});
 textBox(s,"A validated D0 nuclear morphology comparison with a clear go / revise decision for the full study.",310,602,850,46,{size:23,bold:true});
}

for(let i=0;i<deck.slides.items.length;i++){const png=await deck.export({slide:deck.slides.items[i],format:"png",scale:1});await fs.writeFile(path.join(previewDir,`slide-${String(i+1).padStart(2,"0")}.png`),new Uint8Array(await png.arrayBuffer()));}
const montage=await deck.export({format:"webp",montage:true});await fs.writeFile(path.join(buildDir,"deck-montage-v13.webp"),new Uint8Array(await montage.arrayBuffer()));

const { finalizePresentation }=await import(pathToFileURL(path.join(SKILL_DIR,"container_tools","artifact_tool_utils.mjs")).href);
const stagingDir=path.join(buildDir,".codex-finalizer-v13");await fs.mkdir(stagingDir,{recursive:true});const candidatePath=path.join(stagingDir,"candidate.pptx");await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({explicitTotalSlideCount:14,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],workspaceDir,candidatePath,finalPath:FINAL_PPTX,pythonExecutable:RUNTIME_PYTHON,integrityValidatorPath:path.join(SKILL_DIR,"container_tools","inspect_presentation_package_integrity.py"),layoutValidatorPath:path.join(SKILL_DIR,"container_tools","inspect_presentation_layout_geometry.py"),layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit"],fontPolicy:{basis:"design",families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(stagingDir,"validation.json")});
console.log(FINAL_PPTX);
