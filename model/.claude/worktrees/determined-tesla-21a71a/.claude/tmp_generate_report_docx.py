from pathlib import Path
from datetime import datetime, timezone
import zipfile
from xml.sax.saxutils import escape

OUTPUT = Path(r"C:/Users/86191/Desktop/h2-fcu-modern-dashboard/氢燃料电池监控平台论文式报告.docx")

SECTIONS = [
    ("title", "基于实时通信与智能诊断的氢燃料电池监控平台设计与实现"),
    ("heading1", "摘要"),
    ("body", "针对氢燃料电池控制单元在运行监测、故障诊断和远程控制方面存在的数据分散、可视化不足以及智能分析能力有限等问题，本文设计并实现了一套集前端可视化监控、后端实时通信桥接和模型驱动诊断分析于一体的现代化监控平台。系统总体采用分层架构：前端基于 React、TypeScript 与 Vite 构建工业监控界面，后端基于 Python、WebSocket 与 CAN 总线驱动实现设备数据采集与控制指令转发，模型子系统则围绕燃料电池状态分类任务建立了数据预处理、训练、微调、评估与预测的完整流程。"),
    ("body", "在系统实现方面，平台前端可对燃料电池运行参数、故障状态和诊断结果进行实时展示，并支持控制指令下发与日志记录；后端支持真实硬件模式与虚拟仿真模式切换，便于联调与演示；模型模块采用增强型时空图注意力网络进行状态识别，在已有实验结果中取得较高分类精度。项目形成了从设备数据接入、状态可视化、故障分析到控制反馈的闭环能力，可为氢燃料电池系统的运维监控和智能诊断提供工程化支撑。"),
    ("body", "关键词：氢燃料电池；实时监控；CAN 总线；WebSocket；故障诊断；深度学习"),
    ("heading1", "1 引言"),
    ("body", "随着氢能技术的持续发展，燃料电池系统在工业装备、能源动力与智能控制场景中的应用逐步增加。燃料电池控制单元承担着运行调度、状态监测和故障反馈等关键职责，由于运行过程涉及电压、电流、温度、风机、阀门等多类信号，其控制与监测过程具有实时性强、数据维度多、状态关联复杂等特点，传统单一仪表式监测方式已经难以满足复杂工程场景下的可视化与智能化需求。"),
    ("body", "从工程实践看，当前问题主要体现在三个方面：其一，底层 CAN 总线数据与上层可视化界面之间缺乏统一桥接机制，导致设备联调效率较低；其二，运行状态与故障信息虽可采集，但缺少结构化展示与历史追踪能力；其三，已有诊断流程更多依赖经验判断，缺乏模型辅助分析，难以提升异常识别效率与一致性。基于上述问题，本文围绕当前项目构建了一套面向氢燃料电池场景的现代化监控平台。"),
    ("heading1", "2 系统总体设计"),
    ("heading2", "2.1 总体架构"),
    ("body", "本项目采用前端展示层、后端通信层与模型分析层三层架构设计。前端负责构建工业化监控界面，展示燃料电池系统运行参数、状态趋势、故障信息和模型诊断结果，并提供必要的控制交互入口；后端作为核心桥接层，通过 ControlCAN.dll 与 CAN 硬件通信，采集 FCU 报文并解析为结构化状态，再通过 WebSocket 将数据持续推送至前端，同时接收前端控制命令并重新封装为 CAN 控制报文下发到底层设备；模型层围绕燃料电池状态分类任务，提供数据预处理、训练、微调和预测评估等功能。"),
    ("heading2", "2.2 工作流程"),
    ("body", "系统运行时，底层设备通过 CAN 总线产生运行数据，后端解析后生成统一的状态对象；随后通过 WebSocket 以固定频率广播给前端；前端接收后完成实时图表更新、故障展示与交互控制；若启用诊断模块，后端可在广播过程中结合当前状态调用诊断逻辑生成故障类别与置信度结果，并同步发送至前端显示。与此同时，前端也可将人工控制命令或诊断反馈通过 WebSocket 回传后端，实现监控与控制闭环。"),
    ("heading2", "2.3 技术路线"),
    ("body", "项目在技术选型上兼顾了可维护性与工程落地性。前端采用 React 19、TypeScript、Vite 与 Recharts 构建实时工业可视化界面；后端采用 Python、asyncio 与 websockets 构建异步通信服务，并通过 ctypes 调用 Windows 环境下的 CAN 驱动；模型部分采用 TensorFlow/Keras 体系实现状态分类网络。该技术路线使平台具备较好的开发效率、可扩展性与多模块协同能力。"),
    ("heading1", "3 前端监控平台设计"),
    ("heading2", "3.1 前端模块组成"),
    ("body", "前端以 App.tsx 为总控入口，承担状态管理、通信生命周期管理、图表历史缓存、故障日志更新及页面布局组织等职责。系统围绕共享数据契约 types.ts 构建统一的数据接口，保证运行状态、诊断结果、控制参数和图表数据在各组件之间传递一致。组件层面包括工业示意图、趋势图、诊断面板、故障告警抽屉、指标卡片和控制面板等模块。"),
    ("heading2", "3.2 实时可视化功能"),
    ("body", "系统通过 WebSocket 持续接收后端状态广播，并将其转化为页面中的实时运行指标。图表模块支持对关键参数进行滚动式时间序列展示，使操作者能够观察参数变化趋势。界面还提供连接状态提示、故障高亮提示以及在线/离线状态展示，增强运行感知能力。前端同时具备本地 CSV 数据记录能力，便于后续离线分析。"),
    ("heading2", "3.3 前端交互与控制设计"),
    ("body", "前端不仅承担展示职责，还提供模式切换、指令下发和人工反馈等交互功能。控制参数经由统一处理函数更新后，通过 WebSocket 服务发送至后端，再由后端完成 CAN 封装与下发。对于诊断结果，前端允许用户提交反馈信息，从而为后续模型修正与离线训练积累样本。"),
    ("heading1", "4 后端通信与控制系统设计"),
    ("heading2", "4.1 后端总体职责"),
    ("body", "backend/server.py 是后端核心程序，负责初始化运行环境、选择 CAN 通信模式、启动异步任务、管理 WebSocket 客户端连接，并完成数据接收、解析、广播与控制指令转发等工作。backend/config.py 统一管理运行模式、设备类型、CAN 波特率、通道号、WebSocket 地址与广播频率等参数，降低了部署与调试复杂度。"),
    ("heading2", "4.2 CAN 与 WebSocket 桥接机制"),
    ("body", "系统支持 virtual 与 zlg 两种模式。virtual 模式不依赖真实硬件，由后端自动生成模拟 CAN 数据，适合功能演示与前后端联调；zlg 模式通过 ControlCAN.dll 调用真实 USB-CAN 设备接口，实现对燃料电池控制单元的数据采集与指令发送。系统接收到 CAN 报文后，依据协议解析规则映射为结构化状态对象 MachineState，再通过 WebSocket 周期性广播给所有前端客户端。"),
    ("heading2", "4.3 控制与诊断反馈闭环"),
    ("body", "当用户在前端进行控制操作时，前端将控制数据封装为 WebSocket 消息发送给后端，后端根据协议定义生成控制 CAN 报文并调用发送接口写入总线，从而实现对设备行为的远程控制。除控制外，系统还支持诊断反馈消息回传，用户可根据前端显示的诊断结果提交人工修正信息，后端将其保存为反馈样本，为后续模型迭代提供数据基础。"),
    ("heading1", "5 智能诊断与模型子系统设计"),
    ("heading2", "5.1 模型任务定义"),
    ("body", "model 目录构成项目中的算法分析子系统，主要面向燃料电池状态分类任务。结合现有文件与结果可知，该任务以运行特征作为输入，以状态类别作为输出，属于监督式多分类问题。已有元数据表明标签列为“类型”，类别数为 3，说明模型针对多种燃料电池运行状态进行了区分建模。"),
    ("heading2", "5.2 数据预处理流程"),
    ("body", "模型训练前，项目通过 preprocess_utils.py 完成统一预处理流程，包括数据文件加载、特征筛选、无效列删除、缺失值填补、异常值裁剪、标签编码和标准化处理等步骤。同时，系统结合特征重要度进行特征选择，并显式保留关键电流类特征。预处理后生成的元数据会被持久化保存，从而保证训练、微调和预测阶段使用一致的数据处理方案。"),
    ("heading2", "5.3 模型结构与训练流程"),
    ("body", "model.py 实现了增强型 MSTGAT 分类网络，主体结构融合了特征变换、序列建模、图结构邻接关系与注意力机制，整体目标是更充分地提取燃料电池运行数据中的时序关联与特征交互关系。项目还设计了学习率调度与自定义优化器，并提供完整训练、已有模型微调以及评估/预测脚本，形成了较完善的模型生命周期支持。"),
    ("heading2", "5.4 模型结果分析"),
    ("body", "从当前保存结果看，模型在不同训练测试划分下均取得较高性能。例如，results_testdata_80_20 中准确率达到 0.9973、加权 F1 值达到 0.9973；results_testdata_70_30 中准确率达到 0.9982、加权 F1 值达到 0.9982。这表明模型在现有数据集上具有较强分类能力，但后续仍需结合更多真实工况与统一实验规范开展进一步验证。"),
    ("heading1", "6 项目实现特点与工程价值"),
    ("body", "与单纯界面项目或单纯算法项目相比，本项目的特点在于实现了多层系统协同：一是前后端数据链路完整，从 CAN 数据采集到 WebSocket 推送再到页面展示形成闭环；二是支持真实与仿真双模式，便于联调与演示；三是集成诊断分析能力，在传统监控基础上加入模型推理与人工反馈机制；四是兼顾展示与实验，可同时作为工业监控平台和模型验证平台。该项目可服务于燃料电池设备的调试、运行监测和异常分析场景，具有较强的综合实践价值。"),
    ("heading1", "7 存在问题与后续优化方向"),
    ("body", "从进一步工程化角度看，项目仍有若干可优化方向：其一，前端总控逻辑仍较集中，后续可继续加强模块拆分与 hooks 复用；其二，backend/can_protocol.py 与 backend/can_protocol1.py 并存，正式交付前应统一协议实现与文档说明；其三，模型在线集成深度仍可加强，需进一步建立统一的在线诊断与离线评估闭环；其四，实验结果目录存在输出内容差异，说明实验版本管理仍需强化。"),
    ("heading1", "8 结论"),
    ("body", "本文围绕当前项目文件，对一个面向氢燃料电池场景的现代化监控平台进行了系统化整理。项目通过前端可视化、后端通信桥接和模型分析三部分协同，实现了燃料电池状态监测、控制交互和智能诊断的集成设计。系统在架构上具备清晰分层，在实现上兼顾工程联调与模型实验，在结果上表现出较好的可行性与应用潜力。后续若进一步完善模块化设计、统一协议规范并加强在线诊断验证，其工程价值和展示效果还可继续提升。"),
    ("heading1", "参考项目文件"),
    ("body", "App.tsx；index.tsx；types.ts；services/websocketService.ts；services/appwrite.ts；backend/server.py；backend/config.py；backend/can_protocol1.py；backend/diagnosis.py；model/model.py；model/preprocess_utils.py；model/fine_tune_existing_model.py；model/evaluate_or_predict_model.py。"),
]


def build_run(text: str, *, bold: bool = False, size: int = 24, font: str = "宋体") -> str:
    props = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>'
    ]
    if bold:
        props.append('<w:b/>')
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def build_paragraph(text: str, kind: str) -> str:
    if kind == "title":
        ppr = '<w:jc w:val="center"/><w:spacing w:after="240"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{build_run(text, bold=True, size=32)}</w:p>'
    if kind == "heading1":
        ppr = '<w:spacing w:before="200" w:after="120"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{build_run(text, bold=True, size=28)}</w:p>'
    if kind == "heading2":
        ppr = '<w:spacing w:before="120" w:after="80"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{build_run(text, bold=True, size=26)}</w:p>'
    ppr = '<w:jc w:val="both"/><w:spacing w:after="120"/><w:ind w:firstLine="420"/>'
    return f'<w:p><w:pPr>{ppr}</w:pPr>{build_run(text, size=24)}</w:p>'


paragraphs = "".join(build_paragraph(text, kind) for kind, text in SECTIONS)

document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
    xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
    xmlns:o="urn:schemas-microsoft-com:office:office"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
    xmlns:v="urn:schemas-microsoft-com:vml"
    xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
    xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    xmlns:w10="urn:schemas-microsoft-com:office:word"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
    xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
    xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
    xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
    xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
    mc:Ignorable="w14 wp14">
  <w:body>
    {paragraphs}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="851" w:footer="992" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>氢燃料电池监控平台论文式报告</dc:title>
  <dc:creator>Claude Code</dc:creator>
  <cp:lastModifiedBy>Claude Code</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''

app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Claude Code</Application>
</Properties>'''

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types)
    zf.writestr("_rels/.rels", root_rels)
    zf.writestr("docProps/core.xml", core_xml)
    zf.writestr("docProps/app.xml", app_xml)
    zf.writestr("word/document.xml", document_xml)

print(OUTPUT)