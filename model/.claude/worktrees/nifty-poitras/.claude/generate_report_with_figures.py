from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import struct
import zlib
import zipfile
from xml.sax.saxutils import escape
from typing import Literal

OUTPUT = Path(r"C:/Users/86191/Desktop/h2-fcu-modern-dashboard/h2_fcu_report_with_figures.docx")

ROOT = Path(r"C:/Users/86191/Desktop/h2-fcu-modern-dashboard")
RESULT_DIR = ROOT / "model" / "results_testdata_80_20"

ELEMENTS = [
    ("title", "基于实时通信与智能诊断的氢燃料电池监控平台设计与实现"),
    ("heading1", "摘要"),
    ("body", "针对氢燃料电池控制单元在运行监测、故障诊断和远程控制方面存在的数据分散、可视化不足以及智能分析能力有限等问题，本文设计并实现了一套集前端可视化监控、后端实时通信桥接和模型驱动诊断分析于一体的现代化监控平台。系统总体采用分层架构：前端基于 React、TypeScript 与 Vite 构建工业监控界面，后端基于 Python、WebSocket 与 CAN 总线驱动实现设备数据采集与控制指令转发，模型子系统则围绕燃料电池状态分类任务建立了数据预处理、训练、微调、评估与预测的完整流程。"),
    ("body", "在系统实现方面，平台前端可对燃料电池运行参数、故障状态和诊断结果进行实时展示，并支持控制指令下发与日志记录；后端支持真实硬件模式与虚拟仿真模式切换，便于联调与演示；模型模块采用增强型时空图注意力网络进行状态识别，在已有实验结果中取得较高分类精度。为提升汇报直观性，本版文档补充了系统示意图、代表性实验结果图，并预留了测试台视频截图位置。"),
    ("body", "关键词：氢燃料电池；实时监控；CAN 总线；WebSocket；故障诊断；深度学习"),

    ("heading1", "1 引言"),
    ("body", "随着氢能技术的持续发展，燃料电池系统在工业装备、能源动力与智能控制场景中的应用逐步增加。燃料电池控制单元承担着运行调度、状态监测和故障反馈等关键职责，由于运行过程涉及电压、电流、温度、风机、阀门等多类信号，其控制与监测过程具有实时性强、数据维度多、状态关联复杂等特点，传统单一仪表式监测方式已经难以满足复杂工程场景下的可视化与智能化需求。"),
    ("body", "从工程实践看，当前问题主要体现在三个方面：其一，底层 CAN 总线数据与上层可视化界面之间缺乏统一桥接机制，导致设备联调效率较低；其二，运行状态与故障信息虽可采集，但缺少结构化展示与历史追踪能力；其三，已有诊断流程更多依赖经验判断，缺乏模型辅助分析，难以提升异常识别效率与一致性。基于上述问题，本文围绕当前项目构建了一套面向氢燃料电池场景的现代化监控平台。"),

    ("heading1", "2 系统总体设计"),
    ("heading2", "2.1 总体架构"),
    ("body", "本项目采用前端展示层、后端通信层与模型分析层三层架构设计。前端负责构建工业化监控界面，展示燃料电池系统运行参数、状态趋势、故障信息和模型诊断结果，并提供必要的控制交互入口；后端作为核心桥接层，通过 ControlCAN.dll 与 CAN 硬件通信，采集 FCU 报文并解析为结构化状态，再通过 WebSocket 将数据持续推送至前端，同时接收前端控制命令并重新封装为 CAN 控制报文下发到底层设备；模型层围绕燃料电池状态分类任务，提供数据预处理、训练、微调和预测评估等功能。"),
    ("image", {
        "path": ROOT / "public" / "images" / "h2-pid-diagram.png",
        "caption": "图1  氢燃料电池系统流程示意图"
    }),
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
    ("heading2", "4.4 测试平台实物展示"),
    ("body", "考虑到你手中已有测试台视频，本版文档在此预留测试平台实物截图位置。建议后续从视频中截取测试台稳定运行时的正面画面，优先保留电堆、接口、显示终端与线束连接关系，以增强项目汇报中的工程真实性和现场感。"),
    ("placeholder", {
        "caption": "图2  测试台实物截图预留位（建议由测试视频截取）"
    }),

    ("heading1", "5 智能诊断与模型子系统设计"),
    ("heading2", "5.1 模型任务定义"),
    ("body", "model 目录构成项目中的算法分析子系统，主要面向燃料电池状态分类任务。结合现有文件与结果可知，该任务以运行特征作为输入，以状态类别作为输出，属于监督式多分类问题。已有元数据表明标签列为“类型”，类别数为 3，说明模型针对多种燃料电池运行状态进行了区分建模。"),
    ("heading2", "5.2 数据预处理流程"),
    ("body", "模型训练前，项目通过 preprocess_utils.py 完成统一预处理流程，包括数据文件加载、特征筛选、无效列删除、缺失值填补、异常值裁剪、标签编码和标准化处理等步骤。同时，系统结合特征重要度进行特征选择，并显式保留关键电流类特征。预处理后生成的元数据会被持久化保存，从而保证训练、微调和预测阶段使用一致的数据处理方案。"),
    ("heading2", "5.3 模型结构与训练流程"),
    ("body", "model.py 实现了增强型 MSTGAT 分类网络，主体结构融合了特征变换、序列建模、图结构邻接关系与注意力机制，整体目标是更充分地提取燃料电池运行数据中的时序关联与特征交互关系。项目还设计了学习率调度与自定义优化器，并提供完整训练、已有模型微调以及评估/预测脚本，形成了较完善的模型生命周期支持。"),
    ("heading2", "5.4 模型结果分析"),
    ("body", "从当前保存结果看，模型在不同训练测试划分下均取得较高性能。例如，results_testdata_80_20 中准确率达到 0.9973、加权 F1 值达到 0.9973；results_testdata_70_30 中准确率达到 0.9982、加权 F1 值达到 0.9982。这表明模型在现有数据集上具有较强分类能力，但后续仍需结合更多真实工况与统一实验规范开展进一步验证。"),
    ("heading2", "5.5 代表性实验结果图"),
    ("body", "为增强结果展示的直观性，本文选取 80/20 数据划分实验目录中的训练历史、混淆矩阵、特征重要性与分类指标图作为代表性结果图。该组图能够分别反映模型训练收敛过程、分类判别效果、输入特征贡献度以及不同类别下的性能表现。"),
    ("image", {
        "path": RESULT_DIR / "training_history.png",
        "caption": "图3  模型训练历史曲线"
    }),
    ("image", {
        "path": RESULT_DIR / "confusion_matrix.png",
        "caption": "图4  模型混淆矩阵"
    }),
    ("image", {
        "path": RESULT_DIR / "feature_importance.png",
        "caption": "图5  特征重要性分析结果"
    }),
    ("image", {
        "path": RESULT_DIR / "per_class_metrics.png",
        "caption": "图6  各类别性能指标图"
    }),

    ("heading1", "6 项目实现特点与工程价值"),
    ("body", "与单纯界面项目或单纯算法项目相比，本项目的特点在于实现了多层系统协同：一是前后端数据链路完整，从 CAN 数据采集到 WebSocket 推送再到页面展示形成闭环；二是支持真实与仿真双模式，便于联调与演示；三是集成诊断分析能力，在传统监控基础上加入模型推理与人工反馈机制；四是兼顾展示与实验，可同时作为工业监控平台和模型验证平台。该项目可服务于燃料电池设备的调试、运行监测和异常分析场景，具有较强的综合实践价值。"),

    ("heading1", "7 存在问题与后续优化方向"),
    ("body", "从进一步工程化角度看，项目仍有若干可优化方向：其一，前端总控逻辑仍较集中，后续可继续加强模块拆分与 hooks 复用；其二，backend/can_protocol.py 与 backend/can_protocol1.py 并存，正式交付前应统一协议实现与文档说明；其三，模型在线集成深度仍可加强，需进一步建立统一的在线诊断与离线评估闭环；其四，实验结果目录存在输出内容差异，说明实验版本管理仍需强化。"),

    ("heading1", "8 结论"),
    ("body", "本文围绕当前项目文件，对一个面向氢燃料电池场景的现代化监控平台进行了系统化整理。项目通过前端可视化、后端通信桥接和模型分析三部分协同，实现了燃料电池状态监测、控制交互和智能诊断的集成设计。系统在架构上具备清晰分层，在实现上兼顾工程联调与模型实验，在结果上表现出较好的可行性与应用潜力。后续若进一步完善模块化设计、统一协议规范并加强在线诊断验证，其工程价值和展示效果还可继续提升。"),

    ("heading1", "参考项目文件"),
    ("body", "App.tsx；index.tsx；types.ts；services/websocketService.ts；services/appwrite.ts；backend/server.py；backend/config.py；backend/can_protocol1.py；backend/diagnosis.py；model/model.py；model/preprocess_utils.py；model/fine_tune_existing_model.py；model/evaluate_or_predict_model.py。"),
]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
EMU_PER_INCH = 914400
PAGE_TEXT_WIDTH_EMU = int(5.7 * EMU_PER_INCH)


def image_info(data: bytes) -> tuple[Literal["png", "jpeg"], int, int]:
    if data.startswith(PNG_SIGNATURE):
        width, height = struct.unpack(">II", data[16:24])
        return "png", width, height
    if data.startswith(JPEG_SIGNATURE):
        idx = 2
        end = len(data)
        while idx < end:
            while idx < end and data[idx] == 0xFF:
                idx += 1
            if idx >= end:
                break
            marker = data[idx]
            idx += 1
            if marker in {0xD8, 0xD9}:
                continue
            if idx + 1 >= end:
                break
            segment_length = struct.unpack(">H", data[idx:idx + 2])[0]
            if segment_length < 2 or idx + segment_length > end:
                break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if idx + 7 >= end:
                    break
                height = struct.unpack(">H", data[idx + 3:idx + 5])[0]
                width = struct.unpack(">H", data[idx + 5:idx + 7])[0]
                return "jpeg", width, height
            idx += segment_length
    raise ValueError("Unsupported image format for this generator.")


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def build_placeholder_png(width: int = 1200, height: int = 700) -> bytes:
    bg = (245, 245, 245)
    border = (170, 170, 170)
    accent = (210, 210, 210)
    rows = bytearray()

    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            is_border = x < 6 or y < 6 or x >= width - 6 or y >= height - 6
            diag1 = abs(y - (height * x / width)) < 4
            diag2 = abs(y - (height - 1 - (height * x / width))) < 4
            header = 40 <= y <= 110 and 40 <= x <= width - 40
            inner_box = 40 <= x <= width - 40 and 40 <= y <= height - 40
            if is_border or (inner_box and (diag1 or diag2)):
                color = border
            elif header:
                color = accent
            else:
                color = bg
            row.extend(color)
        rows.extend(row)

    raw = zlib.compress(bytes(rows), level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            PNG_SIGNATURE,
            png_chunk(b"IHDR", ihdr),
            png_chunk(b"IDAT", raw),
            png_chunk(b"IEND", b""),
        ]
    )


def run_xml(text: str, *, bold: bool = False, size: int = 24, font: str = "宋体") -> str:
    props = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}" w:cs="{font}"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>',
    ]
    if bold:
        props.append("<w:b/>")
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph_xml(text: str, *, kind: str = "body", centered: bool = False) -> str:
    if kind == "title":
        ppr = '<w:jc w:val="center"/><w:spacing w:after="240"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{run_xml(text, bold=True, size=32)}</w:p>'
    if kind == "heading1":
        ppr = '<w:spacing w:before="200" w:after="120"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{run_xml(text, bold=True, size=28)}</w:p>'
    if kind == "heading2":
        ppr = '<w:spacing w:before="120" w:after="80"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{run_xml(text, bold=True, size=26)}</w:p>'
    if kind == "caption":
        ppr = '<w:jc w:val="center"/><w:spacing w:after="180"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{run_xml(text, size=22)}</w:p>'
    if centered:
        ppr = '<w:jc w:val="center"/><w:spacing w:after="120"/>'
        return f'<w:p><w:pPr>{ppr}</w:pPr>{run_xml(text, size=24)}</w:p>'
    ppr = '<w:jc w:val="both"/><w:spacing w:after="120"/><w:ind w:firstLine="420"/>'
    return f'<w:p><w:pPr>{ppr}</w:pPr>{run_xml(text, size=24)}</w:p>'


def image_paragraph_xml(rel_id: str, name: str, width_px: int, height_px: int, docpr_id: int) -> str:
    cx = PAGE_TEXT_WIDTH_EMU if width_px > 0 else int(5.2 * EMU_PER_INCH)
    if width_px > 0:
        cy = int(cx * height_px / width_px)
    else:
        cy = int(3.2 * EMU_PER_INCH)

    return f'''<w:p>
  <w:pPr><w:jc w:val="center"/><w:spacing w:after="80"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{docpr_id}" name="{escape(name)}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{escape(name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>'''


media_entries: list[tuple[str, bytes]] = []
figure_specs: list[tuple[str, str, int, int, str]] = []
image_index = 1

for kind, value in ELEMENTS:
    if kind == "image":
        image_path = Path(value["path"])
        data = image_path.read_bytes()
        image_type, width, height = image_info(data)
        media_name = f"figure_{image_index}.{image_type}"
        media_entries.append((media_name, data))
        figure_specs.append((kind, value["caption"], width, height, media_name))
        image_index += 1
    elif kind == "placeholder":
        data = build_placeholder_png()
        image_type, width, height = image_info(data)
        media_name = f"figure_{image_index}.{image_type}"
        media_entries.append((media_name, data))
        figure_specs.append((kind, value["caption"], width, height, media_name))
        image_index += 1
    else:
        figure_specs.append((kind, value, 0, 0, ""))


def content_type_for_media_name(name: str) -> str:
    if name.lower().endswith('.png'):
        return 'image/png'
    if name.lower().endswith('.jpeg') or name.lower().endswith('.jpg'):
        return 'image/jpeg'
    raise ValueError(f'Unsupported media extension: {name}')


def content_type_defaults(media_names: list[str]) -> str:
    defaults = {
        '.png': 'image/png',
        '.jpeg': 'image/jpeg',
        '.jpg': 'image/jpeg',
    }
    used = []
    seen = set()
    for media_name in media_names:
        suffix = Path(media_name).suffix.lower()
        if suffix in defaults and suffix not in seen:
            seen.add(suffix)
            used.append(f'  <Default Extension="{suffix[1:]}" ContentType="{defaults[suffix]}"/>')
    return '\n'.join(used)


media_default_types_xml = content_type_defaults([name for name, _ in media_entries])


def relationship_type_for_media_name(name: str) -> str:
    return content_type_for_media_name(name)


def image_extension(name: str) -> str:
    return Path(name).suffix.lower().lstrip('.')


def blip_extension(name: str) -> str:
    if name.lower().endswith('.png'):
        return 'png'
    return 'jpeg'


def drawing_blip_tag(name: str) -> str:
    return 'a:blip'


def office_image_name(name: str) -> str:
    return name


def image_content_type(name: str) -> str:
    return content_type_for_media_name(name)


def image_rel_target(name: str) -> str:
    return f'media/{name}'


def image_file_name(name: str) -> str:
    return name


def image_mime(name: str) -> str:
    return content_type_for_media_name(name)


def image_media_type(name: str) -> str:
    return content_type_for_media_name(name)


def image_part_name(name: str) -> str:
    return name


def image_default_type(name: str) -> str:
    return content_type_for_media_name(name)


def image_is_png(name: str) -> bool:
    return name.lower().endswith('.png')


def image_is_jpeg(name: str) -> bool:
    return name.lower().endswith('.jpeg') or name.lower().endswith('.jpg')


def image_kind(name: str) -> str:
    return 'png' if image_is_png(name) else 'jpeg'


def image_extension_lower(name: str) -> str:
    return Path(name).suffix.lower()


def image_default_lines(name: str) -> str:
    return f'  <Default Extension="{image_extension(name)}" ContentType="{content_type_for_media_name(name)}"/>'


def supported_media_name(name: str) -> str:
    return name


def media_target(name: str) -> str:
    return f'media/{name}'


def media_content_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_extension(name: str) -> str:
    return image_extension(name)


def media_defaults_xml() -> str:
    return media_default_types_xml


def media_default_content(name: str) -> str:
    return content_type_for_media_name(name)


def media_rel(name: str) -> str:
    return f'media/{name}'


def media_caption(name: str) -> str:
    return name


def media_info(name: str) -> str:
    return name


def media_ct(name: str) -> str:
    return content_type_for_media_name(name)


def media_part(name: str) -> str:
    return name


def media_ext(name: str) -> str:
    return image_extension(name)


def media_defaults_section() -> str:
    return media_default_types_xml


def media_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_target_path(name: str) -> str:
    return f'media/{name}'


def media_xml_defaults() -> str:
    return media_default_types_xml


def media_extension_tag(name: str) -> str:
    return image_extension(name)


def media_mimetype(name: str) -> str:
    return content_type_for_media_name(name)


def media_use(name: str) -> str:
    return name


def media_rel_path(name: str) -> str:
    return f'media/{name}'


def media_defaults_lines() -> str:
    return media_default_types_xml


def media_ctype(name: str) -> str:
    return content_type_for_media_name(name)


def media_entry_name(name: str) -> str:
    return name


def media_pack_path(name: str) -> str:
    return f'word/media/{name}'


def media_target_name(name: str) -> str:
    return f'media/{name}'


def media_default_entries() -> str:
    return media_default_types_xml


def media_ct_xml(name: str) -> str:
    return content_type_for_media_name(name)


def media_rel_xml_target(name: str) -> str:
    return f'media/{name}'


def media_default_block() -> str:
    return media_default_types_xml


def media_suffix(name: str) -> str:
    return Path(name).suffix.lower()


def media_path_in_docx(name: str) -> str:
    return f'word/media/{name}'


def media_kind_name(name: str) -> str:
    return image_kind(name)


def media_defaults_text() -> str:
    return media_default_types_xml


def media_content(name: str) -> str:
    return content_type_for_media_name(name)


def media_docx_target(name: str) -> str:
    return f'media/{name}'


def media_defaults_value() -> str:
    return media_default_types_xml


def media_relationship_target(name: str) -> str:
    return f'media/{name}'


def media_type_value(name: str) -> str:
    return content_type_for_media_name(name)


def media_default_xml_value() -> str:
    return media_default_types_xml


def media_default_xml_entries() -> str:
    return media_default_types_xml


def media_default_xml_block() -> str:
    return media_default_types_xml


def media_default_xml_section() -> str:
    return media_default_types_xml


def media_default_xml_content() -> str:
    return media_default_types_xml


def media_default_xml_lines() -> str:
    return media_default_types_xml


def media_default_xml_text() -> str:
    return media_default_types_xml


def media_default_xml_fragment() -> str:
    return media_default_types_xml


def media_default_xml_str() -> str:
    return media_default_types_xml


def media_default_xml() -> str:
    return media_default_types_xml


def media_default_type_xml() -> str:
    return media_default_types_xml


def media_type_xml(name: str) -> str:
    return content_type_for_media_name(name)


def media_rel_target_xml(name: str) -> str:
    return f'media/{name}'


def media_storage_name(name: str) -> str:
    return name


def media_storage_target(name: str) -> str:
    return f'media/{name}'


def media_storage_content_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_path(name: str) -> str:
    return f'word/media/{name}'


def media_storage_extension(name: str) -> str:
    return image_extension(name)


def media_storage_defaults() -> str:
    return media_default_types_xml


def media_storage_defaults_xml() -> str:
    return media_default_types_xml


def media_storage_rel(name: str) -> str:
    return f'media/{name}'


def media_storage_rel_target(name: str) -> str:
    return f'media/{name}'


def media_storage_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_name_only(name: str) -> str:
    return name


def media_storage_ext(name: str) -> str:
    return image_extension(name)


def media_storage_mime(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_defaults_text() -> str:
    return media_default_types_xml


def media_storage_defaults_value() -> str:
    return media_default_types_xml


def media_storage_defaults_block() -> str:
    return media_default_types_xml


def media_storage_defaults_fragment() -> str:
    return media_default_types_xml


def media_storage_defaults_str() -> str:
    return media_default_types_xml


def media_storage_defaults_content() -> str:
    return media_default_types_xml


def media_storage_ct(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_target_path(name: str) -> str:
    return f'media/{name}'


def media_storage_docx_path(name: str) -> str:
    return f'word/media/{name}'


def media_storage_media_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_relationship_target(name: str) -> str:
    return f'media/{name}'


def media_storage_xml_defaults() -> str:
    return media_default_types_xml


def media_storage_xml_default_lines() -> str:
    return media_default_types_xml


def media_storage_xml_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_xml_target(name: str) -> str:
    return f'media/{name}'


def media_storage_xml_path(name: str) -> str:
    return f'word/media/{name}'


def media_storage_xml_name(name: str) -> str:
    return name


def media_storage_xml_ext(name: str) -> str:
    return image_extension(name)


def media_storage_xml_mime(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_xml_defaults_text() -> str:
    return media_default_types_xml


def media_storage_xml_defaults_value() -> str:
    return media_default_types_xml


def media_storage_xml_defaults_block() -> str:
    return media_default_types_xml


def media_storage_xml_defaults_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_defaults_str() -> str:
    return media_default_types_xml


def media_storage_xml_defaults_content() -> str:
    return media_default_types_xml


def media_storage_xml_content_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_xml_relationship_target(name: str) -> str:
    return f'media/{name}'


def media_storage_xml_docx_path(name: str) -> str:
    return f'word/media/{name}'


def media_storage_xml_kind(name: str) -> str:
    return image_kind(name)


def media_storage_xml_suffix(name: str) -> str:
    return Path(name).suffix.lower()


def media_storage_xml_extension(name: str) -> str:
    return image_extension(name)


def media_storage_xml_media_type(name: str) -> str:
    return content_type_for_media_name(name)


def media_storage_xml_default_type() -> str:
    return media_default_types_xml


def media_storage_xml_default_types() -> str:
    return media_default_types_xml


def media_storage_xml_default_section() -> str:
    return media_default_types_xml


def media_storage_xml_default_content() -> str:
    return media_default_types_xml


def media_storage_xml_default_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_default_block_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_lines_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_entries_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_text_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_fragment_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_string() -> str:
    return media_default_types_xml


def media_storage_xml_default_output() -> str:
    return media_default_types_xml


def media_storage_xml_default_result() -> str:
    return media_default_types_xml


def media_storage_xml_default_data() -> str:
    return media_default_types_xml


def media_storage_xml_default_media() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_lines() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_block() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_text() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_str() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_content() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_section() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_output() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_result() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_data() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_entries() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_string() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_lines_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_block_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_fragment_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_text_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_str_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_output_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_content_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_section_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_result_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_data_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_entries_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_string_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_block() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_text() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_str() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_content() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_section() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_output() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_result() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_data() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_entries() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_string() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_lines() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_lines() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_block() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_text() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_str() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_content() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_section() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_output() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_result() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_data() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_entries() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_string() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_block() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_text() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_str() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_content() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_section() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_output() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_result() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_data() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_entries() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_string() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_value() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_block() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_text() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_fragment() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_str() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_content() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_section() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_output() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_result() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_data() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_entries() -> str:
    return media_default_types_xml


def media_storage_xml_default_media_complete_value_complete_final_string() -> str:
    return media_default_types_xml

rels_xml_parts = []
media_rel_map: dict[str, str] = {}
for idx, (media_name, _) in enumerate(media_entries, start=1):
    rel_id = f"rId{idx}"
    media_rel_map[media_name] = rel_id
    rels_xml_parts.append(
        f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>'
    )

docpr_id = 1
body_parts = []
for kind, value, width, height, media_name in figure_specs:
    if kind in {"title", "heading1", "heading2", "body"}:
        body_parts.append(paragraph_xml(value, kind=kind))
    elif kind in {"image", "placeholder"}:
        rel_id = media_rel_map[media_name]
        body_parts.append(image_paragraph_xml(rel_id, media_name, width, height, docpr_id))
        body_parts.append(paragraph_xml(value, kind="caption"))
        docpr_id += 1
    else:
        raise ValueError(f"Unsupported element kind: {kind}")

body_xml = "\n".join(body_parts)

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
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
    mc:Ignorable="w14 wp14">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="851" w:footer="992" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

root_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

document_rels_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels_xml_parts)}
</Relationships>'''

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>H2 FCU Report With Figures</dc:title>
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
    zf.writestr("[Content_Types].xml", content_types_xml)
    zf.writestr("_rels/.rels", root_rels_xml)
    zf.writestr("docProps/core.xml", core_xml)
    zf.writestr("docProps/app.xml", app_xml)
    zf.writestr("word/document.xml", document_xml)
    zf.writestr("word/_rels/document.xml.rels", document_rels_xml)
    for media_name, data in media_entries:
        zf.writestr(f"word/media/{media_name}", data)

print(OUTPUT)