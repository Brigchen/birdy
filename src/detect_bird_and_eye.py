"""
多阶段鸟类检测脚本：
1. 读取图片 EXIF GPS，定位拍摄省市
2. 使用 YOLOv8x-seg 检测鸟体
3. （可选）使用 birdeye.pt 检测鸟眼
4. 使用 ResNet34（本地）或豆包API 进行物种识别
   - 本地模型：ResNet34（10964 类）- 快速、离线
   - 豆包API：在线识别 - 准确率更高、支持全球物种
5. 结合地理位置辅助优化物种识别结果
6. 从 bird_classification.json 补全四级中文分类（目/科/属/种）
7. 裁剪图按「目_科_属_种_省_市_序号」命名，保存至「目/科/属/种」目录结构

支持 RAW 格式文件，使用内置缩略图进行检测
支持混合识别：本地模型与豆包API切换或自动回退

作者: brigchen@gmail.com
版权说明: 基于开源协议，请勿商用
"""
import os
import re
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from ultralytics import YOLO

# 尝试导入rawpy用于RAW文件处理
try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    _RAWPY_AVAILABLE = False
    print("警告: rawpy未安装，RAW文件处理功能受限。安装命令: pip install rawpy")

# 地理编码模块（地名 → GPS 坐标 → EXIF）
try:
    from geo_encoder import (
        geocode_location,
        batch_write_gps_exif,
        write_gps_exif,
        read_gps_exif,
        wgs84_to_gcj02,
    )
    _GEO_ENCODER_AVAILABLE = True
except ImportError:
    _GEO_ENCODER_AVAILABLE = False
    wgs84_to_gcj02 = None  # type: ignore
    print("警告: geo_encoder 模块未找到或依赖缺失（geopy, piexif）")

# 豆包API（可选）
try:
    from doubao_bird_api import DoubaoBirdAPIClient, HybridBirdClassifier
    _BAIDU_API_AVAILABLE = True
except ImportError:
    _BAIDU_API_AVAILABLE = False
    print("提示: 豆包API支持未安装。可用命令安装: pip install requests")


# ─────────────────────────────────────────────────────────────
# 分类学辅助：属名 → (目, 科) 映射表
# 覆盖常见属，未收录的退回 ("未知目", "未知科")
# 来源：IOC World Bird List v14 精简版
# ─────────────────────────────────────────────────────────────
_GENUS_TAXONOMY: Dict[str, Tuple[str, str]] = {
    # 鸵鸟目
    "Struthio": ("鸵鸟目", "鸵鸟科"),
    # 美洲鸵目
    "Rhea": ("美洲鸵目", "美洲鸵科"),
    "Pterocnemia": ("美洲鸵目", "美洲鸵科"),
    # 几维目
    "Apteryx": ("几维目", "几维科"),
    # 鹤鸵目
    "Casuarius": ("鹤鸵目", "鹤鸵科"),
    "Dromaius": ("鹤鸵目", "鸸鹋科"),
    # 凤头鸊鷉目/鸊鷉目
    "Tachybaptus": ("鸊鷉目", "鸊鷉科"),
    "Podiceps": ("鸊鷉目", "鸊鷉科"),
    "Podilymbus": ("鸊鷉目", "鸊鷉科"),
    "Aechmophorus": ("鸊鷉目", "鸊鷉科"),
    # 鸽形目
    "Columba": ("鸽形目", "鸠鸽科"),
    "Streptopelia": ("鸽形目", "鸠鸽科"),
    "Spilopelia": ("鸽形目", "鸠鸽科"),
    "Geopelia": ("鸽形目", "鸠鸽科"),
    "Treron": ("鸽形目", "鸠鸽科"),
    "Ducula": ("鸽形目", "鸠鸽科"),
    "Ptilinopus": ("鸽形目", "鸠鸽科"),
    "Zenaida": ("鸽形目", "鸠鸽科"),
    "Patagioenas": ("鸽形目", "鸠鸽科"),
    # 夜鹰目
    "Caprimulgus": ("夜鹰目", "夜鹰科"),
    "Antrostomus": ("夜鹰目", "夜鹰科"),
    "Chordeiles": ("夜鹰目", "夜鹰科"),
    "Podargus": ("夜鹰目", "蛙口夜鹰科"),
    # 雨燕目
    "Apus": ("雨燕目", "雨燕科"),
    "Chaetura": ("雨燕目", "雨燕科"),
    "Aerodramus": ("雨燕目", "雨燕科"),
    "Collocalia": ("雨燕目", "雨燕科"),
    "Hirundapus": ("雨燕目", "雨燕科"),
    # 蜂鸟目
    "Trochilus": ("蜂鸟目", "蜂鸟科"),
    "Archilochus": ("蜂鸟目", "蜂鸟科"),
    "Calypte": ("蜂鸟目", "蜂鸟科"),
    "Selasphorus": ("蜂鸟目", "蜂鸟科"),
    "Amazilia": ("蜂鸟目", "蜂鸟科"),
    "Phaethornis": ("蜂鸟目", "蜂鸟科"),
    # 鹃形目
    "Cuculus": ("鹃形目", "杜鹃科"),
    "Clamator": ("鹃形目", "杜鹃科"),
    "Hierococcyx": ("鹃形目", "杜鹃科"),
    "Cacomantis": ("鹃形目", "杜鹃科"),
    "Chrysococcyx": ("鹃形目", "杜鹃科"),
    "Coccyzus": ("鹃形目", "杜鹃科"),
    "Geococcyx": ("鹃形目", "杜鹃科"),
    "Centropus": ("鹃形目", "鸦鹃科"),
    # 秧鸡目
    "Rallus": ("秧鸡目", "秧鸡科"),
    "Gallinula": ("秧鸡目", "秧鸡科"),
    "Fulica": ("秧鸡目", "秧鸡科"),
    "Porphyrio": ("秧鸡目", "秧鸡科"),
    "Amaurornis": ("秧鸡目", "秧鸡科"),
    "Porzana": ("秧鸡目", "秧鸡科"),
    "Zapornia": ("秧鸡目", "秧鸡科"),
    "Coturnicops": ("秧鸡目", "秧鸡科"),
    # 鹤形目
    "Grus": ("鹤形目", "鹤科"),
    "Antigone": ("鹤形目", "鹤科"),
    "Balearica": ("鹤形目", "鹤科"),
    "Leucogeranus": ("鹤形目", "鹤科"),
    "Bugeranus": ("鹤形目", "鹤科"),
    # 鸻形目
    "Charadrius": ("鸻形目", "鸻科"),
    "Pluvialis": ("鸻形目", "鸻科"),
    "Vanellus": ("鸻形目", "鸻科"),
    "Hoplopterus": ("鸻形目", "鸻科"),
    "Tringa": ("鸻形目", "鹬科"),
    "Calidris": ("鸻形目", "鹬科"),
    "Gallinago": ("鸻形目", "鹬科"),
    "Scolopax": ("鸻形目", "鹬科"),
    "Numenius": ("鸻形目", "鹬科"),
    "Limosa": ("鸻形目", "鹬科"),
    "Philomachus": ("鸻形目", "鹬科"),
    "Actitis": ("鸻形目", "鹬科"),
    "Xenus": ("鸻形目", "鹬科"),
    "Phalaropus": ("鸻形目", "鹬科"),
    "Haematopus": ("鸻形目", "蛎鹬科"),
    "Recurvirostra": ("鸻形目", "反嘴鹬科"),
    "Himantopus": ("鸻形目", "反嘴鹬科"),
    "Larus": ("鸻形目", "鸥科"),
    "Chroicocephalus": ("鸻形目", "鸥科"),
    "Ichthyaetus": ("鸻形目", "鸥科"),
    "Hydrocoloeus": ("鸻形目", "鸥科"),
    "Hydroprogne": ("鸻形目", "鸥科"),
    "Sterna": ("鸻形目", "鸥科"),
    "Chlidonias": ("鸻形目", "鸥科"),
    "Thalasseus": ("鸻形目", "鸥科"),
    "Rynchops": ("鸻形目", "剪嘴鸥科"),
    "Fratercula": ("鸻形目", "海雀科"),
    "Alca": ("鸻形目", "海雀科"),
    "Uria": ("鸻形目", "海雀科"),
    "Alle": ("鸻形目", "海雀科"),
    "Cepphus": ("鸻形目", "海雀科"),
    # 鹳形目
    "Ciconia": ("鹳形目", "鹳科"),
    "Ephippiorhynchus": ("鹳形目", "鹳科"),
    "Leptoptilos": ("鹳形目", "鹳科"),
    "Mycteria": ("鹳形目", "鹳科"),
    # 鲣鸟目
    "Sula": ("鲣鸟目", "鲣鸟科"),
    "Morus": ("鲣鸟目", "鲣鸟科"),
    "Phalacrocorax": ("鲣鸟目", "鸬鹚科"),
    "Microcarbo": ("鲣鸟目", "鸬鹚科"),
    "Nannopterum": ("鲣鸟目", "鸬鹚科"),
    "Anhinga": ("鲣鸟目", "蛇鹈科"),
    "Fregata": ("鲣鸟目", "军舰鸟科"),
    # 鹈形目
    "Pelecanus": ("鹈形目", "鹈鹕科"),
    "Ardea": ("鹈形目", "鹭科"),
    "Egretta": ("鹈形目", "鹭科"),
    "Casmerodius": ("鹈形目", "鹭科"),
    "Bubulcus": ("鹈形目", "鹭科"),
    "Butorides": ("鹈形目", "鹭科"),
    "Nycticorax": ("鹈形目", "鹭科"),
    "Gorsachius": ("鹈形目", "鹭科"),
    "Ixobrychus": ("鹈形目", "鹭科"),
    "Botaurus": ("鹈形目", "鹭科"),
    "Ardeola": ("鹈形目", "鹭科"),
    "Cochlearius": ("鹈形目", "鹭科"),
    "Threskiornis": ("鹈形目", "鹮科"),
    "Platalea": ("鹈形目", "鹮科"),
    "Plegadis": ("鹈形目", "鹮科"),
    "Eudocimus": ("鹈形目", "鹮科"),
    "Geronticus": ("鹈形目", "鹮科"),
    # 火烈鸟目
    "Phoenicopterus": ("火烈鸟目", "火烈鸟科"),
    # 雁形目
    "Anas": ("雁形目", "鸭科"),
    "Aythya": ("雁形目", "鸭科"),
    "Anser": ("雁形目", "鸭科"),
    "Branta": ("雁形目", "鸭科"),
    "Chen": ("雁形目", "鸭科"),
    "Cygnus": ("雁形目", "鸭科"),
    "Cairina": ("雁形目", "鸭科"),
    "Aix": ("雁形目", "鸭科"),
    "Bucephala": ("雁形目", "鸭科"),
    "Mergus": ("雁形目", "鸭科"),
    "Lophodytes": ("雁形目", "鸭科"),
    "Oxyura": ("雁形目", "鸭科"),
    "Somateria": ("雁形目", "鸭科"),
    "Melanitta": ("雁形目", "鸭科"),
    "Clangula": ("雁形目", "鸭科"),
    "Netta": ("雁形目", "鸭科"),
    "Tadorna": ("雁形目", "鸭科"),
    "Dendrocygna": ("雁形目", "鸭科"),
    "Spatula": ("雁形目", "鸭科"),
    "Mareca": ("雁形目", "鸭科"),
    "Sibirionetta": ("雁形目", "鸭科"),
    # 隼形目
    "Falco": ("隼形目", "隼科"),
    "Microhierax": ("隼形目", "隼科"),
    "Polihierax": ("隼形目", "隼科"),
    # 鹰形目
    "Accipiter": ("鹰形目", "鹰科"),
    "Buteo": ("鹰形目", "鹰科"),
    "Circus": ("鹰形目", "鹰科"),
    "Milvus": ("鹰形目", "鹰科"),
    "Elanus": ("鹰形目", "鹰科"),
    "Haliaeetus": ("鹰形目", "鹰科"),
    "Aquila": ("鹰形目", "鹰科"),
    "Hieraaetus": ("鹰形目", "鹰科"),
    "Spizaetus": ("鹰形目", "鹰科"),
    "Nisaetus": ("鹰形目", "鹰科"),
    "Pandion": ("鹰形目", "鹗科"),
    "Spilornis": ("鹰形目", "鹰科"),
    "Pernis": ("鹰形目", "鹰科"),
    "Gyps": ("鹰形目", "鹰科"),
    "Aegypius": ("鹰形目", "鹰科"),
    "Neophron": ("鹰形目", "鹰科"),
    "Sarcogyps": ("鹰形目", "鹰科"),
    # 鸮形目
    "Bubo": ("鸮形目", "鸱鸮科"),
    "Strix": ("鸮形目", "鸱鸮科"),
    "Asio": ("鸮形目", "鸱鸮科"),
    "Otus": ("鸮形目", "鸱鸮科"),
    "Glaucidium": ("鸮形目", "鸱鸮科"),
    "Ninox": ("鸮形目", "鸱鸮科"),
    "Athene": ("鸮形目", "鸱鸮科"),
    "Tyto": ("鸮形目", "草鸮科"),
    # 犀鸟目
    "Buceros": ("犀鸟目", "犀鸟科"),
    "Anthracoceros": ("犀鸟目", "犀鸟科"),
    "Aceros": ("犀鸟目", "犀鸟科"),
    "Rhyticeros": ("犀鸟目", "犀鸟科"),
    "Ocyceros": ("犀鸟目", "犀鸟科"),
    # 佛法僧目
    "Alcedo": ("佛法僧目", "翠鸟科"),
    "Ceryle": ("佛法僧目", "翠鸟科"),
    "Megaceryle": ("佛法僧目", "翠鸟科"),
    "Halcyon": ("佛法僧目", "翠鸟科"),
    "Todiramphus": ("佛法僧目", "翠鸟科"),
    "Coracias": ("佛法僧目", "佛法僧科"),
    "Merops": ("佛法僧目", "蜂虎科"),
    # 啄木鸟目
    "Picus": ("啄木鸟目", "啄木鸟科"),
    "Dendrocopos": ("啄木鸟目", "啄木鸟科"),
    "Dryocopus": ("啄木鸟目", "啄木鸟科"),
    "Picoides": ("啄木鸟目", "啄木鸟科"),
    "Melanerpes": ("啄木鸟目", "啄木鸟科"),
    "Colaptes": ("啄木鸟目", "啄木鸟科"),
    "Campephilus": ("啄木鸟目", "啄木鸟科"),
    "Jynx": ("啄木鸟目", "啄木鸟科"),
    "Picumnus": ("啄木鸟目", "啄木鸟科"),
    "Blythipicus": ("啄木鸟目", "啄木鸟科"),
    "Chrysophlegma": ("啄木鸟目", "啄木鸟科"),
    "Yungipicus": ("啄木鸟目", "啄木鸟科"),
    "Leiopicus": ("啄木鸟目", "啄木鸟科"),
    "Piculus": ("啄木鸟目", "啄木鸟科"),
    # 雀形目——鸦科
    "Corvus": ("雀形目", "鸦科"),
    "Pica": ("雀形目", "鸦科"),
    "Garrulus": ("雀形目", "鸦科"),
    "Cyanopica": ("雀形目", "鸦科"),
    "Nucifraga": ("雀形目", "鸦科"),
    "Pyrrhocorax": ("雀形目", "鸦科"),
    "Perisoreus": ("雀形目", "鸦科"),
    "Cyanocitta": ("雀形目", "鸦科"),
    "Aphelocoma": ("雀形目", "鸦科"),
    "Gymnorhinus": ("雀形目", "鸦科"),
    # 雀形目——山雀科
    "Parus": ("雀形目", "山雀科"),
    "Cyanistes": ("雀形目", "山雀科"),
    "Periparus": ("雀形目", "山雀科"),
    "Lophophanes": ("雀形目", "山雀科"),
    "Poecile": ("雀形目", "山雀科"),
    "Baeolophus": ("雀形目", "山雀科"),
    "Melaniparus": ("雀形目", "山雀科"),
    "Aegithalos": ("雀形目", "长尾山雀科"),
    # 雀形目——燕科
    "Hirundo": ("雀形目", "燕科"),
    "Delichon": ("雀形目", "燕科"),
    "Cecropis": ("雀形目", "燕科"),
    "Riparia": ("雀形目", "燕科"),
    "Petrochelidon": ("雀形目", "燕科"),
    "Progne": ("雀形目", "燕科"),
    "Tachycineta": ("雀形目", "燕科"),
    "Stelgidopteryx": ("雀形目", "燕科"),
    # 雀形目——莺科/柳莺科
    "Sylvia": ("雀形目", "莺科"),
    "Curruca": ("雀形目", "莺科"),
    "Acrocephalus": ("雀形目", "苇莺科"),
    "Hippolais": ("雀形目", "苇莺科"),
    "Iduna": ("雀形目", "苇莺科"),
    "Locustella": ("雀形目", "蝗莺科"),
    "Phylloscopus": ("雀形目", "柳莺科"),
    "Abrornis": ("雀形目", "柳莺科"),
    "Seicercus": ("雀形目", "柳莺科"),
    "Regulus": ("雀形目", "戴菊科"),
    # 雀形目——鸫科/鹟科
    "Turdus": ("雀形目", "鸫科"),
    "Zoothera": ("雀形目", "鸫科"),
    "Geokichla": ("雀形目", "鸫科"),
    "Ficedula": ("雀形目", "鹟科"),
    "Muscicapa": ("雀形目", "鹟科"),
    "Cyanoptila": ("雀形目", "鹟科"),
    "Cyornis": ("雀形目", "鹟科"),
    "Niltava": ("雀形目", "鹟科"),
    "Eumyias": ("雀形目", "鹟科"),
    "Copsychus": ("雀形目", "鹟科"),
    "Luscinia": ("雀形目", "鹟科"),
    "Calliope": ("雀形目", "鹟科"),
    "Tarsiger": ("雀形目", "鹟科"),
    "Phoenicurus": ("雀形目", "鹟科"),
    "Monticola": ("雀形目", "鹟科"),
    "Saxicola": ("雀形目", "鹟科"),
    "Oenanthe": ("雀形目", "鹟科"),
    # 雀形目——麻雀科/雀科
    "Passer": ("雀形目", "麻雀科"),
    "Montifringilla": ("雀形目", "麻雀科"),
    "Fringilla": ("雀形目", "燕雀科"),
    "Chloris": ("雀形目", "燕雀科"),
    "Carduelis": ("雀形目", "燕雀科"),
    "Spinus": ("雀形目", "燕雀科"),
    "Pyrrhula": ("雀形目", "燕雀科"),
    "Loxia": ("雀形目", "燕雀科"),
    "Carpodacus": ("雀形目", "燕雀科"),
    "Haemorhous": ("雀形目", "燕雀科"),
    "Coccothraustes": ("雀形目", "燕雀科"),
    "Emberiza": ("雀形目", "鹀科"),
    "Schoeniclus": ("雀形目", "鹀科"),
    "Calcarius": ("雀形目", "鹀科"),
    "Melospiza": ("雀形目", "雀科"),
    "Zonotrichia": ("雀形目", "雀科"),
    "Junco": ("雀形目", "雀科"),
    "Spizella": ("雀形目", "雀科"),
    "Passerculus": ("雀形目", "雀科"),
    "Setophaga": ("雀形目", "森莺科"),
    "Dendroica": ("雀形目", "森莺科"),
    "Geothlypis": ("雀形目", "森莺科"),
    "Cardellina": ("雀形目", "森莺科"),
    "Mniotilta": ("雀形目", "森莺科"),
    "Oporornis": ("雀形目", "森莺科"),
    "Vermivora": ("雀形目", "森莺科"),
    "Seiurus": ("雀形目", "森莺科"),
    # 雀形目——椋鸟科
    "Sturnus": ("雀形目", "椋鸟科"),
    "Acridotheres": ("雀形目", "椋鸟科"),
    "Gracula": ("雀形目", "椋鸟科"),
    "Spodiopsar": ("雀形目", "椋鸟科"),
    "Agropsar": ("雀形目", "椋鸟科"),
    "Pastor": ("雀形目", "椋鸟科"),
    "Creatophora": ("雀形目", "椋鸟科"),
    # 雀形目——百灵科
    "Alauda": ("雀形目", "百灵科"),
    "Melanocorypha": ("雀形目", "百灵科"),
    "Calandrella": ("雀形目", "百灵科"),
    "Lullula": ("雀形目", "百灵科"),
    "Eremophila": ("雀形目", "百灵科"),
    "Galerida": ("雀形目", "百灵科"),
    # 雀形目——鹡鸰科
    "Motacilla": ("雀形目", "鹡鸰科"),
    "Anthus": ("雀形目", "鹡鸰科"),
    # 雀形目——伯劳科
    "Lanius": ("雀形目", "伯劳科"),
    # 雀形目——太阳鸟科
    "Nectarinia": ("雀形目", "太阳鸟科"),
    "Cinnyris": ("雀形目", "太阳鸟科"),
    "Aethopyga": ("雀形目", "太阳鸟科"),
    # 雀形目——其他
    "Zosterops": ("雀形目", "绣眼鸟科"),
    "Paradoxornis": ("雀形目", "莺鹛科"),
    "Sitta": ("雀形目", "䴓科"),
    "Certhia": ("雀形目", "旋木雀科"),
    "Troglodytes": ("雀形目", "鹪鹩科"),
    "Cistothorus": ("雀形目", "鹪鹩科"),
    "Mimus": ("雀形目", "嘲鸫科"),
    "Toxostoma": ("雀形目", "嘲鸫科"),
    "Dumetella": ("雀形目", "嘲鸫科"),
    "Melanotis": ("雀形目", "嘲鸫科"),
    "Catbird": ("雀形目", "嘲鸫科"),
    "Campylorhynchus": ("雀形目", "鹪鹩科"),
    "Thryomanes": ("雀形目", "鹪鹩科"),
    "Salpinctes": ("雀形目", "鹪鹩科"),
    "Catharus": ("雀形目", "鸫科"),
    "Hylocichla": ("雀形目", "鸫科"),
    "Sialia": ("雀形目", "鸫科"),
    "Myadestes": ("雀形目", "鸫科"),
    # 雀形目——卷尾/王鹟/扇尾鹟等
    "Dicrurus": ("雀形目", "卷尾科"),
    "Hypothymis": ("雀形目", "王鹟科"),
    "Terpsiphone": ("雀形目", "王鹟科"),
    "Rhipidura": ("雀形目", "扇尾鹟科"),
    # 雀形目——黄鹂科
    "Oriolus": ("雀形目", "黄鹂科"),
    # 雀形目——文鸟/梅花雀
    "Lonchura": ("雀形目", "梅花雀科"),
    "Padda": ("雀形目", "梅花雀科"),
    "Taeniopygia": ("雀形目", "梅花雀科"),
    "Estrilda": ("雀形目", "梅花雀科"),
    "Uraeginthus": ("雀形目", "梅花雀科"),
    # 雀形目——画眉/鹛科
    "Garrulax": ("雀形目", "噪鹛科"),
    "Ianthocincla": ("雀形目", "噪鹛科"),
    "Trochalopteron": ("雀形目", "噪鹛科"),
    "Leiothrix": ("雀形目", "莺鹛科"),
    "Minla": ("雀形目", "莺鹛科"),
    "Alcippe": ("雀形目", "莺鹛科"),
    "Pomatorhinus": ("雀形目", "鹛科"),
    # 雀形目——鸦雀
    "Sinosuthora": ("雀形目", "鸦雀科"),
    "Suthora": ("雀形目", "鸦雀科"),
    "Chleuasicus": ("雀形目", "鸦雀科"),
    # 雀形目——绿鹃/黑头翡翠等
    "Chloropsis": ("雀形目", "叶鹎科"),
    # 雀形目——鹎科
    "Pycnonotus": ("雀形目", "鹎科"),
    "Hypsipetes": ("雀形目", "鹎科"),
    "Ixos": ("雀形目", "鹎科"),
    "Microscelis": ("雀形目", "鹎科"),
    # 雀形目——扇尾莺
    "Cisticola": ("雀形目", "扇尾莺科"),
    "Prinia": ("雀形目", "扇尾莺科"),
    # 雀形目——㛰眉科
    "Urosphena": ("雀形目", "蝗莺科"),
    # 雀形目——鹨
    "Liocichla": ("雀形目", "噪鹛科"),
    # 鸡形目——松鸡/雉/鹑
    "Gallus": ("鸡形目", "雉科"),
    "Phasianus": ("鸡形目", "雉科"),
    "Chrysolophus": ("雀形目", "雉科"),  # 实为鸡形目，修正
    "Perdix": ("鸡形目", "雉科"),
    "Coturnix": ("鸡形目", "雉科"),
    "Arborophila": ("鸡形目", "雉科"),
    "Francolinus": ("鸡形目", "雉科"),
    "Alectoris": ("鸡形目", "雉科"),
    "Tragopan": ("鸡形目", "雉科"),
    "Lophophorus": ("鸡形目", "雉科"),
    "Crossoptilon": ("鸡形目", "雉科"),
    "Catreus": ("鸡形目", "雉科"),
    "Syrmaticus": ("鸡形目", "雉科"),
    "Pucrasia": ("鸡形目", "雉科"),
    "Ithaginis": ("鸡形目", "雉科"),
    "Rheinardia": ("鸡形目", "雉科"),
    "Argusianus": ("鸡形目", "雉科"),
    "Afropavo": ("鸡形目", "雉科"),
    "Pavo": ("鸡形目", "雉科"),
    "Polyplectron": ("鸡形目", "雉科"),
    "Rollulus": ("鸡形目", "雉科"),
    "Melanoperdix": ("鸡形目", "雉科"),
    "Perdicula": ("鸡形目", "雉科"),
    "Ophrysia": ("鸡形目", "雉科"),
    "Excalfactoria": ("鸡形目", "雉科"),
    "Numida": ("鸡形目", "珠鸡科"),
    "Meleagris": ("鸡形目", "吐绶鸡科"),
    "Bonasa": ("鸡形目", "松鸡科"),
    "Tetrao": ("鸡形目", "松鸡科"),
    "Lagopus": ("鸡形目", "松鸡科"),
    "Dendragapus": ("鸡形目", "松鸡科"),
    "Centrocercus": ("鸡形目", "松鸡科"),
    "Tympanuchus": ("鸡形目", "松鸡科"),
}
# 修正鸡形目中误分的条目
_GENUS_TAXONOMY["Chrysolophus"] = ("鸡形目", "雉科")


def get_taxonomy(scientific_name: str) -> Tuple[str, str, str, str]:
    """
    从学名解析分类学信息

    Args:
        scientific_name: 学名，格式 "Genus species" 或 "Genus species subspecies"

    Returns:
        (目, 科, 属, 种) 四元组，未知返回 "未知目"/"未知科"
    """
    parts = scientific_name.strip().split()
    if len(parts) < 2:
        return ("未知目", "未知科", scientific_name, "")
    genus   = parts[0]
    species = parts[1]
    order, family = _GENUS_TAXONOMY.get(genus, ("未知目", "未知科"))
    return (order, family, genus, species)


def sanitize_filename(name: str) -> str:
    """将字符串中不适合用作文件名的字符替换为下划线"""
    return re.sub(r'[\\/:*?"<>|]', '_', name)


# ─────────────────────────────────────────────────────────────
# EXIF GPS 读取：从图片中提取经纬度
# ─────────────────────────────────────────────────────────────

def _dms_tuple_to_decimal_latlon(
    dms, ref
) -> Optional[float]:
    """PIL / 旧版 EXIF 中度分秒 → 十进制度；ref 为 N/S/E/W 字节或 str。"""
    if dms is None:
        return None

    def _frac(v):
        if hasattr(v, "numerator"):
            return v.numerator / v.denominator if v.denominator else 0.0
        if isinstance(v, tuple) and len(v) == 2:
            return v[0] / v[1] if v[1] else 0.0
        return float(v)

    try:
        seq = list(dms)
    except TypeError:
        return None
    if len(seq) < 3:
        return None
    d, m, s = (_frac(x) for x in seq[:3])
    val = d + m / 60.0 + s / 3600.0
    ref_s = ref
    if isinstance(ref_s, bytes):
        ref_s = ref_s.decode(errors="ignore")
    ref_s = (ref_s or "N")[:1].upper()
    if ref_s in ("S", "W"):
        val = -val
    return val


def read_gps_from_xmp(image_path: str) -> Optional[Tuple[float, float]]:
    """
    从文件头附近内嵌的 XMP/XML 中解析 GPS（部分软件仅写入 XMP、未写标准 EXIF GPS IFD）。
    """
    try:
        p = Path(image_path)
        if not p.is_file():
            return None
        size = p.stat().st_size
        if size > 60 * 1024 * 1024:
            return None
        raw = p.read_bytes()[: min(size, 8_000_000)]
        if b"GPSLatitude" not in raw:
            return None
        text = raw.decode("utf-8", errors="ignore")

        def _ref_char(blob: str, is_lat: bool) -> str:
            u = (blob or "").strip().upper()
            for token, ch in (
                ("NORTH", "N"),
                ("SOUTH", "S"),
                ("EAST", "E"),
                ("WEST", "W"),
            ):
                if token in u:
                    return ch
            if u[:1] in ("N", "S", "E", "W"):
                return u[:1]
            return "N" if is_lat else "E"

        def _parse_axis(tag: str, ref_tag: str, is_lat: bool) -> Optional[Tuple[float, str]]:
            pat = rf'(?:exif:|xmpGImg:|<[^:>]*:)?{tag}\s*[^>]*>([^<]+)<'
            m = re.search(pat, text, re.I)
            if not m:
                return None
            body = m.group(1).strip()
            refm = re.search(
                rf'(?:exif:|xmpGImg:|<[^:>]*:)?{ref_tag}\s*[^>]*>([^<]+)<',
                text,
                re.I,
            )
            ref = _ref_char(refm.group(1), is_lat) if refm else ("N" if is_lat else "E")
            dm = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*$", body)
            if dm:
                return float(dm.group(1)), ref
            parts = re.split(r"[,，]\s*", body)
            if len(parts) >= 3:
                d, m_, s = float(parts[0]), float(parts[1]), float(parts[2])
                v = d + m_ / 60.0 + s / 3600.0
                return v, ref
            return None

        la = _parse_axis("GPSLatitude", "GPSLatitudeRef", True)
        lo = _parse_axis("GPSLongitude", "GPSLongitudeRef", False)
        if not la or not lo:
            return None
        lat, lref = la
        lon, wref = lo
        if lref == "S":
            lat = -abs(lat) if lat > 0 else lat
        if wref == "W":
            lon = -abs(lon) if lon > 0 else lon
        if abs(lat) > 90 or abs(lon) > 180:
            return None
        return (lat, lon)
    except Exception:
        return None


def read_gps_from_exif(image_path: str) -> Optional[Tuple[float, float]]:
    """
    从图片 EXIF / XMP 中读取 GPS 经纬度。

    顺序：1) piexif 直读；2) Pillow getexif().get_ifd(GPS)；3) 旧版 _getexif()；
    4) XMP 片段（Adobe 等仅写 XMP 时）。

    使用绝对路径，减少 Windows 下相对路径或符号链接导致的读失败。
    """
    path = str(Path(image_path).expanduser().resolve(strict=False))

    # 1) piexif（与裁剪回写 EXIF 同源，避免仅依赖 PIL 旧 API 导致漏读）
    if _GEO_ENCODER_AVAILABLE:
        try:
            got = read_gps_exif(path, quiet=True)
            if got is not None:
                return (got[0], got[1])
        except Exception:
            pass

    # 2) Pillow 新版 Exif：getexif().get_ifd(IFD.GPS)
    try:
        from PIL import Image as _PIL

        try:
            from PIL.ExifTags import IFD as _IFD

            _gps_ifd_tag = _IFD.GPS
        except Exception:
            _gps_ifd_tag = 0x8825

        with _PIL.open(path) as img:
            exif = img.getexif()
            if exif is not None:
                gps_ifd = exif.get_ifd(_gps_ifd_tag)
                if gps_ifd:
                    lat = _dms_tuple_to_decimal_latlon(
                        gps_ifd.get(2), gps_ifd.get(1, b"N")
                    )
                    lon = _dms_tuple_to_decimal_latlon(
                        gps_ifd.get(4), gps_ifd.get(3, b"E")
                    )
                    if lat is not None and lon is not None:
                        return (lat, lon)
    except Exception:
        pass

    # 3) 旧版 PIL _getexif（部分老环境仍需要）
    try:
        from PIL import Image as _PIL
        import PIL.ExifTags as _Tags

        with _PIL.open(path) as img:
            exif_data = img._getexif()
        if exif_data:
            tag_map = {v: k for k, v in _Tags.TAGS.items()}
            gps_tag_id = tag_map.get("GPSInfo")
            if gps_tag_id is not None and gps_tag_id in exif_data:
                gps_raw = exif_data[gps_tag_id]
                GPS_TAGS = {v: k for k, v in _Tags.GPSTAGS.items()}
                lat_raw = gps_raw.get(GPS_TAGS.get("GPSLatitude", 2))
                lat_ref = gps_raw.get(GPS_TAGS.get("GPSLatitudeRef", 1), "N")
                lon_raw = gps_raw.get(GPS_TAGS.get("GPSLongitude", 4))
                lon_ref = gps_raw.get(GPS_TAGS.get("GPSLongitudeRef", 3), "E")
                if lat_raw is not None and lon_raw is not None:
                    lat = _dms_tuple_to_decimal_latlon(lat_raw, lat_ref)
                    lon = _dms_tuple_to_decimal_latlon(lon_raw, lon_ref)
                    if lat is not None and lon is not None:
                        return (lat, lon)
    except Exception:
        pass

    return read_gps_from_xmp(path)


# ─────────────────────────────────────────────────────────────
# 地理反查：经纬度 → (省, 市)
# 使用自带 GeoJSON 多边形数据，纯 Python point-in-polygon
# ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GEO_ROOT = _PROJECT_ROOT / "data" / "geo"

def _point_in_polygon(lon: float, lat: float, polygon: list) -> bool:
    """
    Ray-casting 算法判断点是否在多边形内。
    polygon: [[lon, lat], ...] 闭合或非闭合均可
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_geojson_feature(lon: float, lat: float, feature: dict) -> bool:
    """判断点是否在 GeoJSON feature 的几何图形内"""
    geom = feature.get("geometry", {})
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        # coords = [outer_ring, *inner_holes]
        outer = coords[0] if coords else []
        return _point_in_polygon(lon, lat, outer)

    elif gtype == "MultiPolygon":
        # coords = [polygon1, polygon2, ...]
        for poly in coords:
            outer = poly[0] if poly else []
            if _point_in_polygon(lon, lat, outer):
                return True
        return False

    return False


# 省级数据缓存（懒加载）
_PROVINCE_FEATURES: Optional[list] = None

def _load_province_features() -> list:
    global _PROVINCE_FEATURES
    if _PROVINCE_FEATURES is None:
        p = _GEO_ROOT / "world" / "中国-area.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _PROVINCE_FEATURES = json.load(f).get("features", [])
        else:
            _PROVINCE_FEATURES = []
    return _PROVINCE_FEATURES


def locate_province(lon: float, lat: float) -> Optional[str]:
    """经纬度 → 省名（找不到返回 None）"""
    for feat in _load_province_features():
        if _point_in_geojson_feature(lon, lat, feat):
            return feat.get("properties", {}).get("name")
    return None


def locate_city(lon: float, lat: float, province: str) -> Optional[str]:
    """
    经纬度 + 省名 → 市名。

    两种情况：
    1. 普通省份：在 world/中国/{省名}/ 目录下遍历各市的 *-area.json
    2. 直辖市（北京/上海/天津/重庆）+ 特别行政区：
       在 world/中国/{省名}-area.json 中查 dname 字段
    """
    china_dir = _GEO_ROOT / "world" / "中国"

    # 优先尝试子目录（普通省份）
    prov_dir = china_dir / province
    if prov_dir.exists():
        for area_file in sorted(prov_dir.glob("*-area.json")):
            try:
                with open(area_file, encoding="utf-8") as f:
                    fc = json.load(f)
                for feat in fc.get("features", []):
                    if _point_in_geojson_feature(lon, lat, feat):
                        dname = feat.get("properties", {}).get("dname")
                        if dname:
                            return dname
            except Exception:
                continue

    # 直辖市/特别行政区：读 {省名}-area.json
    prov_file = china_dir / f"{province}-area.json"
    if prov_file.exists():
        try:
            with open(prov_file, encoding="utf-8") as f:
                fc = json.load(f)
            for feat in fc.get("features", []):
                if _point_in_geojson_feature(lon, lat, feat):
                    props = feat.get("properties", {})
                    # 直辖市的 dname 或 name 即为区/县名，归属市即省本身
                    dname = props.get("dname") or props.get("name")
                    if dname:
                        return dname
        except Exception:
            pass

    return None


def gps_to_location_meta(
    image_path: str,
) -> Tuple[Optional[str], Optional[str], str, Optional[Tuple[float, float]]]:
    """
    读取图片 GPS → 反查省市（含 WGS84/GCJ-02 与经纬顺序回退）。

    Returns:
        (province, city, status, coords)
        - status: \"ok\" | \"no_coords\" | \"no_match\"
        - coords: 从文件读到的经纬度 (lat, lon)（no_coords 时为 None）
    """
    path = str(Path(image_path).expanduser().resolve(strict=False))
    coords = read_gps_from_exif(path)
    if coords is None:
        return (None, None, "no_coords", None)
    lat, lon = coords

    def try_admin(wlat: float, wlon: float) -> Tuple[Optional[str], Optional[str]]:
        pr = locate_province(wlon, wlat)
        if not pr:
            return (None, None)
        ct = locate_city(wlon, wlat, pr)
        return (pr, ct)

    p, c = try_admin(lat, lon)
    if p:
        return (p, c, "ok", (lat, lon))

    if _GEO_ENCODER_AVAILABLE and wgs84_to_gcj02 is not None:
        try:
            glat, glon = wgs84_to_gcj02(lat, lon)
            p, c = try_admin(glat, glon)
            if p:
                return (p, c, "ok", (lat, lon))
        except Exception:
            pass

    p, c = try_admin(lon, lat)
    if p:
        return (p, c, "ok", (lat, lon))

    return (None, None, "no_match", (lat, lon))


def gps_to_location(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """读取图片 GPS → 反查省市。仅要结果时用此函数；需区分失败原因时用 gps_to_location_meta。"""
    p, c, _, _ = gps_to_location_meta(image_path)
    return (p, c)


# ─────────────────────────────────────────────────────────────
# 鸟类分类数据库：bird_classification.json
# 提供学名 → 四级中文分类（目/科/属/种）的精确查询
# ─────────────────────────────────────────────────────────────

_BIRD_CLASSIFICATION_PATH = _PROJECT_ROOT / "data" / "species" / "bird_classification.json"
_GEO_SPECIES_INDEX_PATH = _PROJECT_ROOT / "data" / "species" / "geo_species_index.json"

# 缓存：学名(latin) → {"order_cn","family_cn","genus_cn","species_cn","name_la"}
_SCI_TO_CLASSIFICATION: Optional[Dict[str, Dict]] = None
# 缓存：中文名 → 同上
_CN_TO_CLASSIFICATION:  Optional[Dict[str, Dict]] = None
# 缓存：bird_info 学名/中文名 → index（用于将 API 结果映射进地理约束链）
_SCI_TO_BIRD_INDEX: Optional[Dict[str, int]] = None
_CN_TO_BIRD_INDEX: Optional[Dict[str, int]] = None
# 缓存：中国物种白名单（bird_info index 集合）
_CHINA_SPECIES_INDICES: Optional[set] = None


def _load_bird_classification():
    global _SCI_TO_CLASSIFICATION, _CN_TO_CLASSIFICATION
    if _SCI_TO_CLASSIFICATION is not None:
        return
    _SCI_TO_CLASSIFICATION = {}
    _CN_TO_CLASSIFICATION  = {}

    if not _BIRD_CLASSIFICATION_PATH.exists():
        print(f"警告: bird_classification.json 不存在: {_BIRD_CLASSIFICATION_PATH}")
        return

    with open(_BIRD_CLASSIFICATION_PATH, encoding="utf-8") as f:
        db: Dict[str, Dict] = json.load(f)

    for cn_name, info in db.items():
        entry = {
            "order_cn":  info.get("order_cn",  "未知目"),
            "family_cn": info.get("family_cn", "未知科"),
            "genus_cn":  info.get("genus_cn",  "未知属"),
            "species_cn": cn_name,
            "name_la":   info.get("name_la",   ""),
            "genus_la":  info.get("genus_la",  ""),
        }
        _CN_TO_CLASSIFICATION[cn_name] = entry
        sci = info.get("name_la", "").strip()
        if sci:
            _SCI_TO_CLASSIFICATION[sci] = entry
        # 也用 genus_la 建立属级索引（用于退化匹配）
        genus_la = info.get("genus_la", "").strip()
        if genus_la:
            key_genus = f"__genus__{genus_la}"
            if key_genus not in _SCI_TO_CLASSIFICATION:
                _SCI_TO_CLASSIFICATION[key_genus] = entry


def _load_bird_info_index_maps() -> None:
    """加载 bird_info 的学名/中文名到类别 index 的映射。"""
    global _SCI_TO_BIRD_INDEX, _CN_TO_BIRD_INDEX
    if _SCI_TO_BIRD_INDEX is not None and _CN_TO_BIRD_INDEX is not None:
        return
    _SCI_TO_BIRD_INDEX = {}
    _CN_TO_BIRD_INDEX = {}

    bird_info_path = Path(_BIRD_INFO_PATH)
    if not bird_info_path.exists():
        print(f"警告: bird_info.json 不存在: {bird_info_path}")
        return

    try:
        with open(bird_info_path, encoding="utf-8") as f:
            info_list = json.load(f)
        for idx, row in enumerate(info_list):
            if not isinstance(row, list) or not row:
                continue
            cn = (row[0] if len(row) > 0 else "") or ""
            sci = (row[2] if len(row) > 2 else "") or ""
            cn = str(cn).strip()
            sci = str(sci).strip()
            if sci and sci not in _SCI_TO_BIRD_INDEX:
                _SCI_TO_BIRD_INDEX[sci] = idx
            if cn and cn not in _CN_TO_BIRD_INDEX:
                _CN_TO_BIRD_INDEX[cn] = idx
    except Exception as e:
        print(f"警告: 加载 bird_info 索引映射失败: {e}")


def normalize_api_species_candidates(candidates: List[Dict]) -> List[Dict]:
    """
    规范化 API 候选：
    1) 学名命中分类库时，用标准 species_cn 统一中文名（保留原名到 chinese_name_raw）。
    2) 尝试补齐 index（优先学名，其次中文名），使 API 结果可走强地理约束链。
    """
    if not candidates:
        return candidates

    _load_bird_classification()
    _load_bird_info_index_maps()
    sci_map = _SCI_TO_BIRD_INDEX or {}
    cn_map = _CN_TO_BIRD_INDEX or {}

    normalized: List[Dict] = []
    for c in candidates:
        item = dict(c)
        sci = str(item.get("scientific_name") or "").strip()
        cn = str(item.get("chinese_name") or "").strip()

        # 用学名统一中文显示名（仅在存在标准物种映射时）
        if sci and _SCI_TO_CLASSIFICATION and sci in _SCI_TO_CLASSIFICATION:
            std_cn = (_SCI_TO_CLASSIFICATION[sci].get("species_cn") or "").strip()
            if std_cn:
                if cn and cn != std_cn and "chinese_name_raw" not in item:
                    item["chinese_name_raw"] = cn
                item["chinese_name"] = std_cn

        # 为 API 结果补 index，确保可应用省级/中国强地理过滤
        idx = item.get("index")
        if idx in (None, -1):
            if sci and sci in sci_map:
                item["index"] = sci_map[sci]
            elif cn and cn in cn_map:
                item["index"] = cn_map[cn]

        normalized.append(item)

    return normalized


def _load_china_species_indices() -> set:
    """
    从 geo_species_index.json 加载中国（CN）物种的 bird_info index 集合。
    用于在物种预测时约束候选，过滤掉不在中国分布的物种。
    """
    global _CHINA_SPECIES_INDICES
    if _CHINA_SPECIES_INDICES is not None:
        return _CHINA_SPECIES_INDICES

    _CHINA_SPECIES_INDICES = set()

    if not _GEO_SPECIES_INDEX_PATH.exists():
        print(f"警告: geo_species_index.json 不存在，地理约束功能已禁用")
        return _CHINA_SPECIES_INDICES

    try:
        with open(_GEO_SPECIES_INDEX_PATH, encoding="utf-8") as f:
            geo_data = json.load(f)

        code_mapping: Dict[str, int] = geo_data.get("code_mapping", {})
        regions: Dict[str, Dict]     = geo_data.get("regions", {})

        # 取 CN（中国整体）以及所有 CN-xx 省级区域的并集
        cn_codes: set = set()
        for region_key, region_val in regions.items():
            if region_key == "CN" or region_key.startswith("CN-"):
                sp_codes = region_val.get("species", [])
                cn_codes.update(sp_codes)

        # 将 species_code → bird_info index
        for code in cn_codes:
            idx = code_mapping.get(code)
            if idx is not None:
                _CHINA_SPECIES_INDICES.add(idx)

        print(f"中国物种白名单已加载: {len(_CHINA_SPECIES_INDICES)} 种")
    except Exception as e:
        print(f"警告: 加载中国物种索引失败: {e}")

    return _CHINA_SPECIES_INDICES


# 缓存：geo_species_index 中 CN-xx 区域「展示名」→ bird_info index 集合（用于省级分布约束）
_CN_ADMIN_SPECIES_BY_NAME: Optional[Dict[str, set]] = None


def _normalize_admin_region_name(province: str) -> str:
    """将行政区划全称缩为与 geo_species_index 区域 name 一致的形式（如「上海市」→「上海」）。"""
    if not province:
        return ""
    s = province.strip()
    for suf in (
        "维吾尔自治区",
        "壮族自治区",
        "回族自治区",
        "特别行政区",
        "自治区",
        "省",
        "市",
    ):
        if s.endswith(suf):
            return s[: -len(suf)].strip()
    return s


def _load_cn_admin_species_by_name() -> Dict[str, set]:
    """加载各省级 CN-xx 区域的物种 code → bird_info index，按区域中文名索引。"""
    global _CN_ADMIN_SPECIES_BY_NAME
    if _CN_ADMIN_SPECIES_BY_NAME is not None:
        return _CN_ADMIN_SPECIES_BY_NAME

    _CN_ADMIN_SPECIES_BY_NAME = {}
    if not _GEO_SPECIES_INDEX_PATH.exists():
        return _CN_ADMIN_SPECIES_BY_NAME

    try:
        with open(_GEO_SPECIES_INDEX_PATH, encoding="utf-8") as f:
            geo_data = json.load(f)
        code_mapping: Dict[str, int] = geo_data.get("code_mapping", {})
        regions: Dict[str, Dict] = geo_data.get("regions", {})
        for rkey, rval in regions.items():
            if not (isinstance(rkey, str) and rkey.startswith("CN-") and len(rkey) > 4):
                continue
            disp = (rval.get("name") or "").strip()
            if not disp or disp.startswith("CN-"):
                continue
            acc: set = set()
            for code in rval.get("species", []):
                idx = code_mapping.get(code)
                if idx is not None:
                    acc.add(idx)
            if acc:
                _CN_ADMIN_SPECIES_BY_NAME[disp] = acc
    except Exception as e:
        print(f"警告: 加载省级物种分布索引失败: {e}")

    return _CN_ADMIN_SPECIES_BY_NAME


def _resolve_province_species_set(province: str) -> Optional[set]:
    """根据省/直辖市名解析该省级区域在分布索引中的物种 index 集合。"""
    if not province:
        return None
    m = _load_cn_admin_species_by_name()
    if not m:
        return None
    key = _normalize_admin_region_name(province)
    if key in m:
        return m[key]
    raw = province.strip()
    if raw in m:
        return m[raw]
    return None


# 不在当前地理名单（省/中国分布索引）内时，仅当置信度严格高于该值才保留
_GEO_OUTSIDE_LIST_CONF: float = 0.75

# 参与地理筛选与「地理符合」判定的模型候选数量上限
_GEO_CANDIDATE_TOP_N: int = 10


def _geo_top5_promote_by_province(
    ordered: List[Dict],
    province: Optional[str],
    *,
    max_rank: int = 5,
    max_conf_gap: float = 0.35,
) -> List[Dict]:
    """
    若模型第一名不在当前省份分布内，则在 top max_rank 中选取
    「落在该省物种集合内」且置信度不低于 (第一名 − max_conf_gap) 的候选提到首位。
    """
    if not province or len(ordered) < 2:
        return ordered
    region = _resolve_province_species_set(province)
    if not region:
        return ordered
    top0 = ordered[0]
    if top0.get("index") in region:
        return ordered
    head = ordered[: min(max_rank, len(ordered))]
    top_conf = float(top0.get("confidence") or 0)
    best: Optional[Dict] = None
    best_c = -1.0
    for c in head:
        if c.get("index") not in region:
            continue
        cf = float(c.get("confidence") or 0)
        if cf + max_conf_gap < top_conf:
            continue
        if cf > best_c:
            best = c
            best_c = cf
    if best is None:
        return ordered
    return [best] + [x for x in ordered if x is not best]


def lookup_classification(chinese_name: str, scientific_name: str = "") -> Dict[str, str]:
    """
    查询四级中文分类（目/科/属/种）。

    优先精确匹配学名，其次匹配属名，最后匹配中文名。
    找不到时回退到 get_taxonomy() 的拉丁结果。

    Returns:
        {"order_cn", "family_cn", "genus_cn", "species_cn"}
    """
    _load_bird_classification()

    # 1. 精确学名匹配
    if scientific_name and scientific_name in _SCI_TO_CLASSIFICATION:
        e = _SCI_TO_CLASSIFICATION[scientific_name]
        return {"order_cn": e["order_cn"], "family_cn": e["family_cn"],
                "genus_cn": e["genus_cn"], "species_cn": e["species_cn"]}

    # 2. 属名退化匹配（只用 Genus 部分）
    if scientific_name:
        genus = scientific_name.split()[0]
        key_genus = f"__genus__{genus}"
        if key_genus in _SCI_TO_CLASSIFICATION:
            e = _SCI_TO_CLASSIFICATION[key_genus]
            return {"order_cn": e["order_cn"], "family_cn": e["family_cn"],
                    "genus_cn": e["genus_cn"], "species_cn": chinese_name or e["species_cn"]}

    # 3. 中文名匹配
    if chinese_name and chinese_name in _CN_TO_CLASSIFICATION:
        e = _CN_TO_CLASSIFICATION[chinese_name]
        return {"order_cn": e["order_cn"], "family_cn": e["family_cn"],
                "genus_cn": e["genus_cn"], "species_cn": e["species_cn"]}

    # 4. 回退：用旧 _GENUS_TAXONOMY（仅拉丁名）
    order, family, genus_la, sp_ep = get_taxonomy(scientific_name)
    return {
        "order_cn":   order,
        "family_cn":  family,
        "genus_cn":   genus_la or "未知属",
        "species_cn": chinese_name or "未知种",
    }


# ─────────────────────────────────────────────────────────────
# 地理辅助物种优化：对中国分布索引（geo_species_index）中的类略提高排序权重，
# 避免「只要有白名单命中就丢弃所有非白名单」导致顶一高置信度结果被扔掉。
# ─────────────────────────────────────────────────────────────

def geo_refine_species(
    candidates: List[Dict],
    province: Optional[str],
    city: Optional[str],
    geo_mode: str = "china",
    species_conf_threshold: Optional[float] = None,
) -> List[Dict]:
    """
    利用地理位置对物种候选列表进行过滤与重排序。

    Args:
        candidates:              BirdSpeciesClassifier.predict() 返回的 top-k 列表
                                 每个元素含 "index"（bird_info 序号）、"confidence" 等字段
        province:                省名（可 None，有 GPS 时才有值）
        city:                    市名（可 None）
        geo_mode:                地理约束模式
                                   "china"  → 默认，对中国分布索引中的类略提高排序权重
                                   "auto"   → 有 GPS 时同 province，无 GPS 时同 china
                                   "none"   → 不做地理加权
        species_conf_threshold:  候选置信度下限（GUI「未知种类阈值」启用时传入）。
                                   为 None 时不做 top10 置信度初筛，仅按地理规则筛选。

    Returns:
        过滤 + 重排后的候选列表
        - 附加 geo_location 字段（若有省市信息）
        - 仅取模型置信度前 _GEO_CANDIDATE_TOP_N 名参与地理规则
        - 名单内（当前省或中国分布索引）：若 species_conf_threshold 为 None 则全部参与地理；
          否则须置信度 ≥ species_conf_threshold
        - 名单外：须置信度 > _GEO_OUTSIDE_LIST_CONF（默认 0.75）才保留
        - 若前 N 名中存在「名单内」物种，则结果仅保留名单内候选（否则保留上一步通过阈值的全部）
        - china/province 模式：对结果略提高「在中国分布索引」内的排序权重
    """
    if not candidates:
        return candidates

    original_candidates = list(candidates)

    # ── 附加地理信息字段 ──────────────────────────────────────
    geo_loc = ""
    if province:
        geo_loc = f"{province}{city or ''}"
    for c in candidates:
        if geo_loc:
            c["geo_location"] = geo_loc

    effective_mode = geo_mode
    if geo_mode == "auto":
        effective_mode = "province" if province else "china"

    # ── 仅取 top-N；可选：再按 GUI「未知种类阈值」初筛 ─────────
    head_n = original_candidates[: min(_GEO_CANDIDATE_TOP_N, len(original_candidates))]
    if species_conf_threshold is None:
        filtered = list(head_n)
    else:
        thr = float(species_conf_threshold)
        filtered = [
            c for c in head_n
            if float(c.get("confidence") or 0) >= thr
        ]

    if effective_mode == "none" or not filtered:
        return filtered

    # ── 当前地理「名单」：有省且能解析省级索引则用省；否则用中国并集 ──
    geo_list: Optional[set] = None
    if province:
        geo_list = _resolve_province_species_set(province)
    if not geo_list and effective_mode in ("china", "province"):
        geo_list = _load_china_species_indices()
    if not geo_list:
        return filtered

    def _in_geo_list(c: Dict) -> bool:
        idx = c.get("index")
        if idx is None or idx == -1:
            return False
        return idx in geo_list

    thr_hi = float(_GEO_OUTSIDE_LIST_CONF)
    allowed: List[Dict] = []
    for c in filtered:
        cf = float(c.get("confidence") or 0)
        if _in_geo_list(c):
            allowed.append(c)
        elif cf > thr_hi:
            allowed.append(c)

    geo_compliant = [c for c in allowed if _in_geo_list(c)]
    if geo_compliant:
        filtered = geo_compliant
    else:
        filtered = allowed

    # 有省名时仍可做 top5 内省级微调（名单已多为省内时通常不变）
    if province and effective_mode != "none":
        filtered = _geo_top5_promote_by_province(
            filtered, province, max_rank=_GEO_CANDIDATE_TOP_N
        )

    # ── 中国物种地理约束（轻微排序加分）───────────────────────
    if effective_mode in ("china", "province"):
        china_indices = _load_china_species_indices()
        if china_indices:
            china_set = china_indices
            _GEO_SORT_EPS = 0.005

            def _rank_key(c: Dict) -> float:
                base = float(c.get("confidence") or 0)
                if c.get("index") in china_set:
                    return base + _GEO_SORT_EPS
                return base

            filtered.sort(key=_rank_key, reverse=True)

    return filtered


# ─────────────────────────────────────────────────────────────
# 物种识别模块（ResNet34 + bird_info.json，10964 类）
# ─────────────────────────────────────────────────────────────
_BIRDY_ROOT = _PROJECT_ROOT
_SPECIES_MODEL_PATH = str(_BIRDY_ROOT / "models" / "birdiden_v1.pth")
_BIRD_INFO_PATH = str(_BIRDY_ROOT / "models" / "bird_info.json")
_DEFAULT_BIRD_YOLO = str(_BIRDY_ROOT / "models" / "yolov8x-seg.pt")
_DEFAULT_EYE_YOLO = str(_BIRDY_ROOT / "models" / "birdeye.pt")

# 低于接受阈值的识别结果按「未知」归入 未知目/未知科/未知属/未知
UNKNOWN_SPECIES_CLASSIFICATION = {
    "order_cn": "未知目",
    "family_cn": "未知科",
    "genus_cn": "未知属",
    "species_cn": "未知",
}


def _runtime_error_bad_torch_file(err: Exception, path: str, title_cn: str) -> RuntimeError:
    """将 PyTorch 读权重失败包装为可读中文说明（常见于损坏/截断/LFS 指针）。"""
    raw = str(err).lower()
    if any(
        x in raw
        for x in ("zip archive", "central directory", "pytorchstreamreader", "invalid zip")
    ):
        return RuntimeError(
            f"{title_cn}\n"
            f"路径: {os.path.abspath(path)}\n\n"
            f"常见原因：文件损坏、下载未完成、误把 Git LFS 指针当权重，或模型不在 models 下。\n"
            f"YOLO 的 .pt 与物种 .pth 体积通常较大（至少数十 MB），请核对大小后重新拷贝或下载。\n\n"
            f"原始错误: {err}"
        )
    return RuntimeError(f"{title_cn}\n路径: {os.path.abspath(path)}\n{err}")


class BirdSpeciesClassifier:
    """
    基于 ResNet34 的鸟类物种分类器
    直接加载 state_dict，无需 IPC 服务进程
    """

    def __init__(
        self,
        model_path: str = _SPECIES_MODEL_PATH,
        bird_info_path: str = _BIRD_INFO_PATH,
        device: Optional[str] = None,
    ):
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = torch.device(
            "cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu")
        )

        # 加载物种信息
        with open(bird_info_path, encoding="utf-8") as f:
            self.bird_info: List[List[str]] = json.load(f)

        # 先加载 state_dict 获取实际输出类别数
        try:
            try:
                state_dict = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                # PyTorch < 2.0 无 weights_only
                state_dict = torch.load(model_path, map_location=self.device)
            except Exception:
                # 部分旧 .pth 含非张量 pickle，weights_only=True 会失败
                state_dict = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise _runtime_error_bad_torch_file(
                e, model_path, "物种分类权重（birdiden_v1.pth）无法读取"
            ) from e
        self.num_classes = state_dict["fc.weight"].shape[0]  # 从权重直接获取实际类别数

        # 构建 ResNet34，fc 维度与权重一致
        self.model = models.resnet34(weights=None)
        self.model.fc = torch.nn.Linear(512, self.num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 标准 ImageNet 预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        print(f"物种分类器已加载: {self.num_classes} 类，设备: {self.device}")

    def predict(self, img_bgr: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        对 BGR 格式的 numpy 图像进行物种识别

        Args:
            img_bgr: OpenCV BGR 格式图像（鸟的裁剪区域）
            top_k:   返回置信度最高的 k 个物种

        Returns:
            [{"index": int, "chinese_name": ..., "english_name": ..., "scientific_name": ..., "confidence": float}, ...]
            index 为 bird_info 中的序号，用于地理白名单约束
        """
        import torch

        if img_bgr is None or img_bgr.size == 0:
            return []

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_k = min(top_k, self.num_classes)
        values, indices = torch.topk(probs, top_k)

        results = []
        for conf, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
            if idx < len(self.bird_info):
                info = self.bird_info[idx]
                results.append({
                    "index":          idx,
                    "chinese_name":   info[0],
                    "english_name":   info[1] if len(info) > 1 else "",
                    "scientific_name": info[2] if len(info) > 2 else "",
                    "confidence":     round(conf, 4),
                })
        return results


class BirdAndEyeDetector:
    def __init__(
        self,
        bird_model_path: str = _DEFAULT_BIRD_YOLO,
        eye_model_path: str = _DEFAULT_EYE_YOLO,
        species_model_path: str = _SPECIES_MODEL_PATH,
        bird_info_path: str = _BIRD_INFO_PATH,
        bird_conf: float = 0.5,
        eye_conf: float = 0.25,
        device: str = None,
        enable_species: bool = True,
        enable_eye: bool = False,
        geo_mode: str = "china",
        species_conf: Optional[float] = None,
        doubao_config: Optional[Dict] = None,
        use_local_model: bool = True,
        min_species_accept_confidence: Optional[float] = None,
    ):
        """
        初始化多阶段鸟类检测器

        Args:
            bird_model_path:    鸟类检测模型路径（YOLOv8）
            eye_model_path:     鸟眼检测模型路径
            species_model_path: 物种识别模型路径（ResNet34）
            bird_info_path:     物种信息 JSON 路径
            bird_conf:          鸟类检测置信度阈值（默认 0.5）
            eye_conf:           鸟眼检测置信度阈值
            device:             运行设备 (cuda/cpu)
            enable_species:     是否启用物种识别（False 则跳过，加快速度）
            enable_eye:         是否启用鸟眼检测（默认 False，不影响主检测流程）
            geo_mode:           地理约束模式："china"（默认）| "auto" | "none"
            species_conf:       已弃用别名：与 min_species_accept_confidence 同步（保留仅为兼容旧调用）
            doubao_config:       豆包API配置 {"api_key": ...}
            use_local_model:    是否默认使用本地模型（True）或豆包API（False）
            min_species_accept_confidence: 未知种类阈值（0~1），与 GUI 一致；None 表示不启用：
                不对 top10 做置信度初筛，且不因顶一低于阈值清空结果（仅地理规则 + 名单外>0.75）。
        """
        self.bird_conf = bird_conf
        self.eye_conf = eye_conf
        self.enable_species = enable_species
        self.enable_eye = enable_eye
        self.geo_mode = geo_mode
        self.use_local_model = use_local_model
        if min_species_accept_confidence is None:
            self.min_species_accept_confidence = None
            self.species_conf = None
        else:
            self.min_species_accept_confidence = float(min_species_accept_confidence)
            self.species_conf = self.min_species_accept_confidence
        self.species_method = "unknown"  # 记录使用的识别方法

        # 自动选择设备
        if device is None:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"加载鸟类检测模型: {bird_model_path}")
        try:
            self.bird_model = YOLO(bird_model_path)
        except Exception as e:
            raise _runtime_error_bad_torch_file(
                e, bird_model_path, "鸟类检测模型（yolov8x-seg.pt）无法读取"
            ) from e
        self.bird_model.to(device)

        # 鸟眼检测（可选）
        if self.enable_eye:
            print(f"加载鸟眼检测模型: {eye_model_path}")
            try:
                self.eye_model = YOLO(eye_model_path)
            except Exception as e:
                raise _runtime_error_bad_torch_file(
                    e, eye_model_path, "鸟眼检测模型（birdeye.pt）无法读取"
                ) from e
            self.eye_model.to(device)
        else:
            self.eye_model = None
            print("鸟眼检测已禁用（可用 --eye 参数开启）")

        print(f"使用设备: {device}")

        # 加载物种分类器（支持混合模式）
        self.species_classifier = None
        self.hybrid_classifier = None
        
        if self.enable_species:
            try:
                # 加载本地模型
                self.species_classifier = BirdSpeciesClassifier(
                    model_path=species_model_path,
                    bird_info_path=bird_info_path,
                )
                print(f"✓ 本地物种分类器已加载")
            except Exception as e:
                print(f"警告: 本地物种分类器加载失败: {e}")
                self.species_classifier = None
            
            # 配置豆包API（如果提供）
            if doubao_config and _BAIDU_API_AVAILABLE:
                try:
                    self.hybrid_classifier = HybridBirdClassifier(
                        doubao_config=doubao_config,
                        local_model=self.species_classifier,
                        use_local=use_local_model,
                        fallback_to_online=True,
                    )
                    print(f"✓ 混合分类器已初始化（默认: {'本地模型' if use_local_model else '豆包API'}）")
                except Exception as e:
                    print(f"警告: 混合分类器初始化失败: {e}")
                    self.hybrid_classifier = None
            
            # 检查是否有可用的分类器
            if self.species_classifier is None and self.hybrid_classifier is None:
                print(f"警告: 物种识别功能已禁用（没有可用的分类器）")
                self.enable_species = False

    @staticmethod
    def is_raw_file(file_path: str) -> bool:
        """检查文件是否为RAW格式"""
        raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.raf', '.pef', '.rw2'}
        return Path(file_path).suffix.lower() in raw_extensions

    @staticmethod
    def read_raw_thumbnail(file_path: str) -> np.ndarray:
        """
        读取RAW文件的内置缩略图

        Args:
            file_path: RAW文件路径

        Returns:
            OpenCV格式的图像 (BGR)
        """
        if not _RAWPY_AVAILABLE:
            raise RuntimeError("rawpy未安装，无法处理RAW文件。请运行: pip install rawpy")

        try:
            with rawpy.imread(file_path) as raw:
                # 尝试获取内置缩略图
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        # 解码JPEG缩略图
                        img_array = np.frombuffer(thumb.data, dtype=np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if image is not None:
                            return image
                except Exception as e:
                    print(f"  无法提取缩略图: {e}，尝试使用RAW数据")

                # 如果没有缩略图或提取失败，使用postprocess生成预览图（较低质量，较快）
                rgb = raw.postprocess(
                    half_size=True,  # 一半尺寸，加快处理
                    use_camera_wb=True,  # 使用相机白平衡
                    no_auto_bright=True,  # 不自动调整亮度
                    output_bps=8,  # 8位输出
                )
                # RGB转BGR
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return image

        except Exception as e:
            raise RuntimeError(f"无法读取RAW文件 {file_path}: {e}")

    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """
        加载图片，支持普通格式和RAW格式

        Args:
            file_path: 图片路径

        Returns:
            OpenCV格式的图像 (BGR)
        """
        if BirdAndEyeDetector.is_raw_file(file_path):
            print(f"  检测到RAW文件，使用内置缩略图")
            return BirdAndEyeDetector.read_raw_thumbnail(file_path)
        else:
            # 普通图片格式
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"无法读取图片: {file_path}")
            return image

    def detect_birds(self, image: np.ndarray) -> List[Dict]:
        """
        检测图片中的鸟

        Returns:
            鸟的位置列表，每个元素包含bbox和conf
        """
        results = self.bird_model(image, conf=self.bird_conf, verbose=False)
        birds = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                birds.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "class": cls,
                    "class_name": result.names.get(cls, "bird"),
                })

        return birds

    def detect_eyes_in_crop(
        self, crop_img: np.ndarray, offset_x: int = 0, offset_y: int = 0
    ) -> List[Dict]:
        """
        在裁剪出的鸟区域中检测鸟眼（仅 enable_eye=True 时有效）

        Args:
            crop_img: 裁剪出的鸟区域图像
            offset_x: 在原图中的x偏移量
            offset_y: 在原图中的y偏移量

        Returns:
            鸟眼位置列表（坐标是相对于原图的），未启用时返回空列表
        """
        if not self.enable_eye or self.eye_model is None:
            return []

        results = self.eye_model(crop_img, conf=self.eye_conf, verbose=False)
        eyes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # 转换坐标到原图
                eyes.append({
                    "bbox": [
                        int(x1) + offset_x,
                        int(y1) + offset_y,
                        int(x2) + offset_x,
                        int(y2) + offset_y,
                    ],
                    "conf": conf,
                    "class": cls,
                    "class_name": result.names.get(cls, "BirdEye"),
                })

        return eyes

    def detect(
        self,
        image_path: str,
        manual_province: Optional[str] = None,
        manual_city: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        对单张图片进行多阶段检测（含 GPS 解析 + 物种分类补全）

        Args:
            image_path:     图片路径
            manual_province: 无 EXIF GPS 时使用的省名（如界面地图选点反查结果）
            manual_city:     无 EXIF GPS 时使用的市名

        Returns:
            标注后的图像和检测结果字典
            结果字典新增字段:
              - province: 省名（无 GPS 则 None）
              - city:     市名（无 GPS 则 None）
            每只鸟新增字段:
              - classification: {"order_cn","family_cn","genus_cn","species_cn"}
        """
        # ── GPS 地理定位 ─────────────────────────────────────────
        province, city = (None, None)
        try:
            province, city, gps_st, gps_xy = gps_to_location_meta(image_path)
            if province:
                print(f"GPS 定位: {province}{city or ''}")
            elif gps_st == "no_coords":
                print(
                    "GPS 定位: 未能从 EXIF/XMP 读取 GPS（或文件路径无法访问）"
                )
            elif gps_st == "no_match" and gps_xy:
                print(
                    "GPS 定位: 已读取坐标 "
                    f"({gps_xy[0]:.5f}, {gps_xy[1]:.5f})，"
                    "但未匹配到中国省级边界（境外、海上、或底图与 EXIF 坐标系不一致）"
                )
            else:
                print("GPS 定位: 无 GPS 信息或不在中国境内")
        except Exception as e:
            print(f"GPS 定位失败: {e}")

        if not province and manual_province:
            province = manual_province.strip()
            if manual_city:
                city = manual_city.strip()
            elif not city:
                city = None
            if province:
                print(f"物种地理参考: 使用界面/指定的省市 {province}{city or ''}")

        # 读取图片（支持RAW格式）
        image = self.load_image(image_path)

        original_image = image.copy()
        h, w = image.shape[:2]

        # 第一阶段：检测鸟
        birds = self.detect_birds(image)
        print(f"检测到 {len(birds)} 只鸟")

        all_eyes = []

        # 第二阶段（可选鸟眼） & 第三阶段（物种）：逐只鸟处理
        for i, bird in enumerate(birds):
            x1, y1, x2, y2 = bird["bbox"]

            # 裁剪出鸟的区域（稍微扩大一点范围）
            margin = 10
            crop_x1 = max(0, x1 - margin)
            crop_y1 = max(0, y1 - margin)
            crop_x2 = min(w, x2 + margin)
            crop_y2 = min(h, y2 + margin)

            bird_crop = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

            if bird_crop.size == 0:
                continue

            # 鸟眼检测（可选）
            if self.enable_eye:
                eyes = self.detect_eyes_in_crop(bird_crop, crop_x1, crop_y1)
                print(f"  鸟 {i+1}: 检测到 {len(eyes)} 个鸟眼")
                bird["eyes"] = eyes
                all_eyes.extend(eyes)
            else:
                bird["eyes"] = []


            # 物种识别
            if self.enable_species:
                species_preds = []
                method = "unknown"
                baidu_raw = None
                species_debug: Dict[str, Union[str, int, float, bool, None]] = {
                    "raw_candidate_count": 0,
                    "post_geo_count": 0,
                    "species_threshold_enabled": self.min_species_accept_confidence is not None,
                    "candidate_filter_threshold": (
                        None
                        if self.min_species_accept_confidence is None
                        else float(self.min_species_accept_confidence)
                    ),
                    "min_accept_threshold": (
                        None
                        if self.min_species_accept_confidence is None
                        else float(self.min_species_accept_confidence)
                    ),
                    "raw_top1_name": None,
                    "raw_top1_confidence": None,
                    "post_geo_top1_name": None,
                    "post_geo_top1_confidence": None,
                    "unknown_reason": None,
                }
                
                # 优先使用混合分类器（如果可用）
                if self.hybrid_classifier is not None:
                    geo_hint = "中国"
                    if province:
                        geo_hint = f"{province}{city or ''}"
                    species_preds, method = self.hybrid_classifier.predict(
                        bird_crop, top_k=10, geolocation=geo_hint
                    )
                    self.species_method = method
                    # 检查是否使用了豆包API
                    if method.startswith("豆包"):
                        # 保存豆包API的原始响应
                        bird["doubao_raw"] = "使用豆包API进行识别"
                elif self.species_classifier is not None:
                    # 回退到本地模型
                    species_preds = self.species_classifier.predict(bird_crop, top_k=10)
                    method = "本地模型"
                    self.species_method = method

                # 豆包/API 结果标准化：学名统一中文名 + 补 index 以启用强地理约束
                if species_preds and any(
                    (p.get("api_source") == "doubao") or str(method).startswith("豆包")
                    for p in species_preds
                ):
                    species_preds = normalize_api_species_candidates(species_preds)

                raw_preds = list(species_preds or [])
                species_debug["raw_candidate_count"] = len(raw_preds)
                if raw_preds:
                    species_debug["raw_top1_name"] = (
                        raw_preds[0].get("chinese_name")
                        or raw_preds[0].get("english_name")
                        or raw_preds[0].get("scientific_name")
                        or "未知"
                    )
                    species_debug["raw_top1_confidence"] = float(
                        raw_preds[0].get("confidence") or 0
                    )
                
                if species_preds:
                    # 地理约束 + 置信度过滤（默认中国范围）
                    species_preds = geo_refine_species(
                        species_preds, province, city,
                        geo_mode=self.geo_mode,
                        species_conf_threshold=self.min_species_accept_confidence,
                    )
                    species_debug["post_geo_count"] = len(species_preds)
                    if species_preds:
                        species_debug["post_geo_top1_name"] = (
                            species_preds[0].get("chinese_name")
                            or species_preds[0].get("english_name")
                            or species_preds[0].get("scientific_name")
                            or "未知"
                        )
                        species_debug["post_geo_top1_confidence"] = float(
                            species_preds[0].get("confidence") or 0
                        )
                    top0 = species_preds[0] if species_preds else None
                    skip_low_clear = (
                        top0
                        and top0.get("api_source") == "doubao"
                        and top0.get("subject_type") not in (None, "", "bird")
                    )
                    if (
                        species_preds
                        and not skip_low_clear
                        and self.min_species_accept_confidence is not None
                        and float(species_preds[0].get("confidence") or 0)
                        < float(self.min_species_accept_confidence)
                    ):
                        species_debug["unknown_reason"] = (
                            f"top1_conf_below_min_accept"
                            f" ({float(species_preds[0].get('confidence') or 0):.3f}"
                            f" < {float(self.min_species_accept_confidence):.3f})"
                        )
                        species_preds = []
                else:
                    species_debug["post_geo_count"] = 0

                bird["species"] = species_preds
                bird["species_method"] = method  # 记录使用的识别方法
                bird["species_debug"] = species_debug
                if baidu_raw:
                    bird["baidu_raw"] = baidu_raw
                
                if species_preds:
                    top = species_preds[0]
                    print(f"  鸟 {i+1}: 物种 = {top['chinese_name']} ({top['english_name']}) "
                          f"置信度={top['confidence']:.2%} [{method}]"
                          + (f"  [地理约束: {top.get('geo_location','')}]" if top.get('geo_location') else ""))
                else:
                    if species_debug["unknown_reason"] is None:
                        if int(species_debug["raw_candidate_count"]) == 0:
                            species_debug["unknown_reason"] = "no_model_candidates"
                        elif int(species_debug["post_geo_count"]) == 0:
                            if self.min_species_accept_confidence is not None:
                                species_debug["unknown_reason"] = (
                                    "all_candidates_filtered_below_unknown_kind_threshold"
                                    f" (< {float(self.min_species_accept_confidence):.3f})"
                                )
                            else:
                                species_debug["unknown_reason"] = (
                                    "all_candidates_removed_by_geo_rules"
                                )
                        else:
                            species_debug["unknown_reason"] = "all_candidates_filtered_after_geo_refine"
                    print(
                        f"  鸟 {i+1}: 物种 = 未知 [{method}] | 原始候选={species_debug['raw_candidate_count']}, "
                        f"地理后={species_debug['post_geo_count']}, "
                        f"raw_top1_name={species_debug['raw_top1_name']}, "
                        f"raw_top1_conf={species_debug['raw_top1_confidence']}, "
                        f"geo_top1_name={species_debug['post_geo_top1_name']}, "
                        f"geo_top1_conf={species_debug['post_geo_top1_confidence']}, "
                        f"未知种类阈值="
                        f"{'未启用(仅地理)' if self.min_species_accept_confidence is None else f'{float(self.min_species_accept_confidence):.3f}'}, "
                        f"原因={species_debug['unknown_reason']}"
                    )
            else:
                bird["species"] = []
                bird["species_method"] = "已禁用"

            # 四级中文分类补全（豆包非鸟：用人像/其它动物/其它 + 子标签归档）
            sp_list = bird.get("species", [])
            if sp_list:
                top = sp_list[0]
                if (
                    top.get("api_source") == "doubao"
                    and top.get("subject_type") not in (None, "", "bird")
                ):
                    root = (top.get("archive_root_cn") or "").strip() or "其它"
                    tag = (top.get("archive_tag_cn") or "未分类").strip()
                    disp = (top.get("chinese_name") or tag).strip() or "未命名"
                    bird["classification"] = {
                        "order_cn": root,
                        "family_cn": tag,
                        "genus_cn": "—",
                        "species_cn": disp,
                    }
                else:
                    bird["classification"] = lookup_classification(
                        chinese_name=top.get("chinese_name", ""),
                        scientific_name=top.get("scientific_name", ""),
                    )
            else:
                bird["classification"] = dict(UNKNOWN_SPECIES_CLASSIFICATION)

        # 在图像上画框和标注
        result_image = self.visualize(image, birds)

        results = {
            "birds":       birds,
            "total_birds": len(birds),
            "total_eyes":  len(all_eyes),
            "province":    province,
            "city":        city,
        }

        return result_image, results

    def set_species_model(self, use_local: bool):
        """
        切换物种识别模型
        
        Args:
            use_local: True 使用本地模型，False 使用豆包API
        """
        if self.hybrid_classifier is None:
            print("✗ 混合分类器不可用，无法切换模型")
            return False
        
        self.hybrid_classifier.set_model_mode(use_local)
        self.use_local_model = use_local
        mode_name = "本地模型" if use_local else "豆包API"
        print(f"✓ 已切换物种识别模式: {mode_name}")
        return True
    
    def get_species_method(self) -> str:
        """获取当前使用的物种识别方法"""
        return self.species_method

    def visualize(self, image: np.ndarray, birds: List[Dict]) -> np.ndarray:
        """
        在图像上画出鸟框、鸟眼框，并标注物种信息（中文名 + 置信度）

        Args:
            image: 原始图像（BGR）
            birds: 鸟的检测结果列表（含 eyes、species 字段）

        Returns:
            标注后的图像（BGR）
        """
        from PIL import Image, ImageDraw, ImageFont

        # BGR → RGB，转为 PIL Image
        img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        h, w = image.shape[:2]
        scale = max(w, h) / 1500.0

        # 线宽、字号
        bird_lw   = max(3, int(4 * scale))
        eye_lw    = max(3, int(3 * scale))
        main_size = max(28, int(36 * scale))   # 物种主标签字号
        sub_size  = max(20, int(26 * scale))   # 副标签（置信度/鸟眼）字号
        pad       = max(6, int(8 * scale))     # 标签内边距

        # 颜色（RGB）
        BIRD_CLR  = (50,  210,  50)   # 鲜绿：鸟框
        EYE_CLR   = (255,  50,  50)   # 亮红：鸟眼框
        SPEC_CLR  = (255, 220,   0)   # 金黄：物种标签文字
        DET_CLR   = (50,  210,  50)   # 同鸟框绿：YOLO 置信度文字
        BG_CLR    = (0,    0,    0)   # 黑色背景

        # 尝试加载系统中文字体
        font_main = font_sub = None
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",     # 黑体
            "C:/Windows/Fonts/simsun.ttc",     # 宋体
            "C:/Windows/Fonts/arial.ttf",      # Arial（无中文支持但作为备用）
        ]
        for fp in font_candidates:
            if os.path.exists(fp):
                try:
                    font_main = ImageFont.truetype(fp, main_size)
                    font_sub  = ImageFont.truetype(fp, sub_size)
                    break
                except Exception:
                    continue
        if font_main is None:
            font_main = ImageFont.load_default()
            font_sub  = ImageFont.load_default()

        def draw_rect_outline(draw_obj, xy, color, lw):
            """画带黑色描边的矩形框（增加对比度）"""
            x1, y1, x2, y2 = xy
            # 黑色描边
            for d in range(lw + 2):
                draw_obj.rectangle([x1 - d, y1 - d, x2 + d, y2 + d], outline=(0, 0, 0))
            # 主色框
            for d in range(lw):
                draw_obj.rectangle([x1 - d // 2, y1 - d // 2,
                                     x2 + d // 2, y2 + d // 2], outline=color)

        def draw_label(draw_obj, text, xy, font, text_color, bg_color, pad):
            """在指定坐标画背景色块 + 文字"""
            x, y = xy
            bbox = draw_obj.textbbox((x, y), text, font=font)
            bx1, by1, bx2, by2 = bbox
            # 背景
            draw_obj.rectangle(
                [bx1 - pad, by1 - pad, bx2 + pad, by2 + pad],
                fill=bg_color
            )
            draw_obj.text((x, y), text, font=font, fill=text_color)
            return by2 + pad  # 返回底部 y，便于多行堆叠

        label_gap = max(4, int(6 * scale))

        def estimate_label_height(lines_fonts):
            """估算多行标签总高度（含间距）"""
            total = 0
            for text, font in lines_fonts:
                bbox = draw.textbbox((0, 0), text, font=font)
                total += (bbox[3] - bbox[1]) + pad * 2 + label_gap
            return total

        def pick_label_y(box_y1, box_y2, label_h, img_h):
            """
            返回标签块的起始 y 坐标：
            - 优先放框上方（框外），如果空间够
            - 否则放框下方（框外）
            """
            above_y = box_y1 - label_h - label_gap
            if above_y >= 0:
                return above_y          # 框上方，完全在图片内
            below_y = box_y2 + label_gap
            if below_y + label_h <= img_h:
                return below_y          # 框下方，完全在图片内
            # 两侧都放不下，退而求其次放在框上方（允许贴边裁剪）
            return max(0, above_y)

        def clamp_label_x(x, text, font, img_w):
            """确保标签不超出右侧边界"""
            bbox = draw.textbbox((x, 0), text, font=font)
            over = (bbox[2] + pad) - img_w
            if over > 0:
                x = max(0, x - over)
            return x

        for i, bird in enumerate(birds):
            x1, y1, x2, y2 = bird["bbox"]
            det_conf = bird["conf"]

            # ── 鸟框 ──
            draw_rect_outline(draw, (x1, y1, x2, y2), BIRD_CLR, bird_lw)

            # ── 组装标签行：仅物种名 + 物种置信度 ──────────────
            species = bird.get("species", [])
            if species:
                top = species[0]
                cname   = top["chinese_name"]
                sp_conf = top["confidence"]
                # 仅一行：物种中文名 + 置信度
                lines = [
                    (f"{cname}  {sp_conf:.1%}", font_main, SPEC_CLR),
                ]
            else:
                # 无物种识别结果，只显示鸟体序号
                lines = [(f"Bird {i+1}", font_main, DET_CLR)]

            # ── 计算标签块总高度，选定起始 y ──
            lf_pairs = [(t, f) for t, f, _ in lines]
            label_h  = estimate_label_height(lf_pairs)
            cur_y    = pick_label_y(y1, y2, label_h, h)

            # ── 逐行绘制（框外，左对齐，x 防越界）──
            for text, font, color in lines:
                lx = clamp_label_x(x1, text, font, w)
                cur_y = draw_label(draw, text, (lx, cur_y), font, color, BG_CLR, pad)
                cur_y += label_gap

            # ── 鸟眼框 ──
            for eye in bird.get("eyes", []):
                ex1, ey1, ex2, ey2 = eye["bbox"]
                econf = eye["conf"]
                draw_rect_outline(draw, (ex1, ey1, ex2, ey2), EYE_CLR, eye_lw)
                eye_text = f"Eye {econf:.2f}"
                eye_h = sub_size + pad * 2
                eye_label_y = pick_label_y(ey1, ey2, eye_h, h)
                elx = clamp_label_x(ex1, eye_text, font_sub, w)
                draw_label(draw, eye_text, (elx, eye_label_y),
                           font_sub, EYE_CLR, BG_CLR, pad)

        # PIL RGB → OpenCV BGR
        result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result_bgr

    def crop_species(
        self,
        image: np.ndarray,
        birds: List[Dict],
        output_dir: str,
        source_path: str = "",
        counter: Optional[Dict] = None,
        margin_ratio: float = 1.0,
        province: Optional[str] = None,
        city: Optional[str] = None,
    ) -> List[str]:
        """
        根据鸟体检测框裁剪原图，按分类层级保存。

        目录结构：<output_dir>/<目>/<科>/<属>/<种>/
        文件命名：<目>_<科>_<属>_<种>_<省>_<市>_<序号>.jpg

        裁剪规则：
        - 在每只鸟的检测框四周扩展 margin_ratio 倍框尺寸的边距
        - 同一物种的所有扩展框取并集（最小外接矩形），裁剪为一张图

        Args:
            image:       原始 BGR 图像（未标注）
            birds:       detect() 返回的鸟列表，含 bbox / species / classification 字段
            output_dir:  裁剪根目录（目/科/属/种 自动在此下创建）
            source_path: 原图路径，用于提取拍摄时间（EXIF）
            counter:     全局编号计数器 {"n": int}，跨图片累计；None 则单张从 1 开始
            margin_ratio: 边距倍率，默认 1.0
            province:    省名（来自 GPS 定位，可 None）
            city:        市名（来自 GPS 定位，可 None）

        Returns:
            保存的裁剪图路径列表
        """
        from datetime import datetime as _dt

        os.makedirs(output_dir, exist_ok=True)

        if counter is None:
            counter = {"n": 0}

        # ── 获取拍摄时间 ─────────────────────────────────────────
        photo_time = _dt.now().strftime("%Y%m%d_%H%M%S")
        if source_path:
            try:
                from PIL import Image as _PIL_Image
                with _PIL_Image.open(source_path) as _im:
                    exif = _im._getexif() or {}
                    dt_str = exif.get(36867) or exif.get(306, "")
                    if dt_str:
                        photo_time = dt_str.replace(":", "").replace(" ", "_")[:15]
            except Exception:
                pass

        img_h, img_w = image.shape[:2]

        # ── 按物种分组（key = 学名或中文名）──────────────────────
        species_groups: Dict[str, Dict] = {}

        for bird in birds:
            x1, y1, x2, y2 = bird["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            mx = int(bw * margin_ratio)
            my = int(bh * margin_ratio)
            ex1 = max(0, x1 - mx);  ey1 = max(0, y1 - my)
            ex2 = min(img_w, x2 + mx); ey2 = min(img_h, y2 + my)

            sp_list = bird.get("species", [])
            clf     = bird.get("classification", {})

            if sp_list:
                top = sp_list[0]
                if (
                    top.get("api_source") == "doubao"
                    and top.get("subject_type") not in (None, "", "bird")
                ):
                    root = (top.get("archive_root_cn") or "其它").strip()
                    tag = (top.get("archive_tag_cn") or "未分类").strip()
                    key = f"__nb__|{root}|{tag}"
                    sci = ""
                    cname = (top.get("chinese_name") or tag).strip() or "未知"
                    if not clf:
                        clf = {
                            "order_cn": root,
                            "family_cn": tag,
                            "genus_cn": "—",
                            "species_cn": cname,
                        }
                else:
                    sci = top.get("scientific_name", "").strip()
                    cname = top.get("chinese_name", "未知")
                    key = sci if sci else cname
            else:
                key   = "未知物种"
                cname = "未知"
                sci   = ""
                if not clf:
                    clf = dict(UNKNOWN_SPECIES_CLASSIFICATION)

            if key not in species_groups:
                species_groups[key] = {
                    "scientific_name": sci,
                    "chinese_name":    cname,
                    "classification":  clf,
                    "boxes": [],
                }
            species_groups[key]["boxes"].append((ex1, ey1, ex2, ey2))

        # ── 每物种裁剪一张图 ──────────────────────────────────────
        saved_paths: List[str] = []

        for key, grp in species_groups.items():
            sci_name = grp["scientific_name"]
            cname    = grp["chinese_name"]
            clf      = grp["classification"]
            boxes    = grp["boxes"]

            # 合并扩展框 → 最小外接矩形
            cx1 = max(0,     min(b[0] for b in boxes))
            cy1 = max(0,     min(b[1] for b in boxes))
            cx2 = min(img_w, max(b[2] for b in boxes))
            cy2 = min(img_h, max(b[3] for b in boxes))

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = image[cy1:cy2, cx1:cx2].copy()

            # ── 四级分类名（优先用 classification，否则退化）──────
            order_cn  = clf.get("order_cn",  "") or "未知目"
            family_cn = clf.get("family_cn", "") or "未知科"
            genus_cn  = clf.get("genus_cn",  "") or "未知属"
            species_cn = clf.get("species_cn", "") or cname or "未知"

            # 豆包非鸟：<root>/<人像|其它动物|其它>/<子标签>/（两级 + 扁平文件名）
            non_bird_dir = key.startswith("__nb__|")
            if non_bird_dir:
                save_dir = os.path.join(
                    output_dir,
                    sanitize_filename(order_cn),
                    sanitize_filename(family_cn),
                )
            else:
                save_dir = os.path.join(
                    output_dir,
                    sanitize_filename(order_cn),
                    sanitize_filename(family_cn),
                    sanitize_filename(genus_cn),
                    sanitize_filename(species_cn),
                )
            os.makedirs(save_dir, exist_ok=True)

            counter["n"] += 1
            seq = str(counter["n"]).zfill(5)

            prov_part = sanitize_filename(province) if province else "未知省"
            city_part = sanitize_filename(city)     if city     else "未知市"

            if non_bird_dir:
                fname_parts = [
                    sanitize_filename(order_cn),
                    sanitize_filename(family_cn),
                    prov_part,
                    city_part,
                    seq,
                ]
            else:
                fname_parts = [
                    sanitize_filename(order_cn),
                    sanitize_filename(family_cn),
                    sanitize_filename(genus_cn),
                    sanitize_filename(species_cn),
                    prov_part,
                    city_part,
                    seq,
                ]
            filename = "_".join(fname_parts) + ".jpg"
            out_path = os.path.join(save_dir, filename)

            # ── 保存裁剪图 ──────────────────────────────────────────
            # 先用OpenCV保存，再用PIL保留EXIF
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 如果有原图路径，尝试复制EXIF信息到裁剪图
            if source_path:
                try:
                    import piexif
                    from PIL import Image as PIL_Image
                    
                    # 读取原图EXIF
                    try:
                        original_img = PIL_Image.open(source_path)
                        exif_dict = piexif.load(source_path)
                        
                        # 修改部分EXIF信息
                        if "0th" in exif_dict:
                            # 更新拍摄地点信息
                            exif_dict["GPS"] = exif_dict.get("GPS", {})
                            
                            # 添加裁剪标记
                            if 271 in exif_dict["0th"]:  # Make
                                exif_dict["0th"][271] = b"BirdDetection-Cropped"
                        
                        # 序列化EXIF
                        exif_bytes = piexif.dump(exif_dict)
                        
                        # 用PIL打开、修改EXIF、保存
                        crop_pil = PIL_Image.open(out_path)
                        crop_pil.save(out_path, "JPEG", exif=exif_bytes, quality=95)
                        
                    except Exception as e:
                        # 如果提取EXIF失败，仅保留基本信息
                        pass
                        
                except ImportError:
                    # 如果没有piexif，直接保存
                    pass
            
            saved_paths.append(out_path)
            print(f"  裁剪保存: {filename}")
            print(f"    路径: {save_dir}")
            print(f"    坐标: [{cx1},{cy1},{cx2},{cy2}]  尺寸: {cx2-cx1}×{cy2-cy1}px")
            print(f"    物种: {species_cn}（{sci_name}）  位置: {prov_part}{city_part}")

        return saved_paths

    def copy_original_by_top_species(
        self,
        source_path: str,
        birds: List[Dict],
        output_dir: str,
        province: Optional[str] = None,
        city: Optional[str] = None,
        counter: Optional[Dict] = None,
    ) -> List[str]:
        """
        不裁剪时：将原图复制到「整张图内置信度最高」的物种对应 目/科/属/种 目录。
        无鸟、无物种时归入未知分级；若启用了 min_species_accept_confidence，
        且最高置信度低于该值时也归入未知分级（未启用阈值时仅看是否有物种结果）。
        """
        import shutil

        if counter is None:
            counter = {"n": 0}
        if not source_path or not os.path.isfile(source_path):
            return []

        os.makedirs(output_dir, exist_ok=True)

        best_conf = -1.0
        best_sp: Optional[Dict] = None
        for bird in birds:
            for sp in bird.get("species") or []:
                c = float(sp.get("confidence") or 0)
                if c > best_conf:
                    best_conf = c
                    best_sp = sp

        non_bird = (
            best_sp is not None
            and best_sp.get("api_source") == "doubao"
            and best_sp.get("subject_type") not in (None, "", "bird")
        )

        if non_bird:
            order_cn = (best_sp.get("archive_root_cn") or "其它").strip()
            family_cn = (best_sp.get("archive_tag_cn") or "未分类").strip()
            save_dir = os.path.join(
                output_dir,
                sanitize_filename(order_cn),
                sanitize_filename(family_cn),
            )
        elif best_sp is None or (
            self.min_species_accept_confidence is not None
            and best_conf < float(self.min_species_accept_confidence)
        ):
            clf = dict(UNKNOWN_SPECIES_CLASSIFICATION)
            order_cn = clf.get("order_cn", "") or "未知目"
            family_cn = clf.get("family_cn", "") or "未知科"
            genus_cn = clf.get("genus_cn", "") or "未知属"
            species_cn = clf.get("species_cn", "") or "未知"
            save_dir = os.path.join(
                output_dir,
                sanitize_filename(order_cn),
                sanitize_filename(family_cn),
                sanitize_filename(genus_cn),
                sanitize_filename(species_cn),
            )
        else:
            clf = lookup_classification(
                best_sp.get("chinese_name", ""),
                best_sp.get("scientific_name", ""),
            )
            order_cn = clf.get("order_cn", "") or "未知目"
            family_cn = clf.get("family_cn", "") or "未知科"
            genus_cn = clf.get("genus_cn", "") or "未知属"
            species_cn = clf.get("species_cn", "") or "未知"
            save_dir = os.path.join(
                output_dir,
                sanitize_filename(order_cn),
                sanitize_filename(family_cn),
                sanitize_filename(genus_cn),
                sanitize_filename(species_cn),
            )
        os.makedirs(save_dir, exist_ok=True)

        counter["n"] += 1
        seq = str(counter["n"]).zfill(5)
        prov_part = sanitize_filename(province) if province else "未知省"
        city_part = sanitize_filename(city) if city else "未知市"
        stem, ext = os.path.splitext(os.path.basename(source_path))
        if not ext:
            ext = ".jpg"
        fname = f"{sanitize_filename(stem)}_{prov_part}_{city_part}_{seq}{ext}"
        out_path = os.path.join(save_dir, fname)
        shutil.copy2(source_path, out_path)
        print(f"  原图归档（按顶一物种）: {out_path}")
        return [out_path]


def process_folder(
    input_folder: str,
    output_folder: str,
    bird_model_path: str = str(_BIRDY_ROOT / "models" / "yolov8x-seg.pt"),
    eye_model_path: str = str(_BIRDY_ROOT / "models" / "birdeye.pt"),
    species_model_path: str = _SPECIES_MODEL_PATH,
    bird_info_path: str = _BIRD_INFO_PATH,
    enable_species: bool = True,
    enable_eye: bool = False,
    crop_mode: bool = False,
    crop_dir: str = "",
    margin_ratio: float = 1.0,
    geo_mode: str = "china",
    species_conf: float = 0.5,
    location: Optional[str] = None,
):
    """
    处理整个文件夹的图片

    Args:
        input_folder:       输入图片文件夹
        output_folder:      检测标注结果文件夹
        bird_model_path:    鸟类检测模型路径
        eye_model_path:     鸟眼检测模型路径
        species_model_path: 物种识别模型路径
        bird_info_path:     物种信息 JSON 路径
        enable_species:     是否启用物种识别
        enable_eye:         是否启用鸟眼检测（默认 False）
        crop_mode:          是否同时生成裁剪图
        crop_dir:           裁剪图根目录（空则使用 output_folder/crops）
        margin_ratio:       裁剪边距倍率（默认 1.0 = 100%）
        geo_mode:           地理约束模式："china"（默认）| "auto" | "none"
        species_conf:       未知种类阈值（与 GUI / --species-conf 一致，默认 0.5）
        location:           地理位置名称（如"杭州西湖"），若提供则自动写入GPS EXIF
    """
    os.makedirs(output_folder, exist_ok=True)

    # ── 地理编码：地名 → GPS 坐标 → 写入 EXIF ──────────────────────────
    if location and _GEO_ENCODER_AVAILABLE:
        print(f"\n地理编码: '{location}' → GPS 坐标")
        coords = geocode_location(location)
        if coords:
            print(f"成功: ({coords[0]:.6f}, {coords[1]:.6f})")
            success = batch_write_gps_exif(input_folder, coords[0], coords[1])
            if success > 0:
                print(f"✓ 已为 {success} 张图片写入 GPS EXIF")
            else:
                print("✗ GPS EXIF 写入失败")
        else:
            print(f"✗ 地理编码失败: '{location}'")
    elif location and not _GEO_ENCODER_AVAILABLE:
        print("警告: --location 参数指定但 geo_encoder 模块不可用")
        print("请安装: pip install geopy piexif Pillow")

    # 确定裁剪目录
    if crop_mode:
        _crop_dir = crop_dir if crop_dir else os.path.join(output_folder, "crops")
        os.makedirs(_crop_dir, exist_ok=True)
        crop_counter = {"n": 0}

    # 初始化检测器
    detector = BirdAndEyeDetector(
        bird_model_path=bird_model_path,
        eye_model_path=eye_model_path,
        species_model_path=species_model_path,
        bird_info_path=bird_info_path,
        enable_species=enable_species,
        enable_eye=enable_eye,
        geo_mode=geo_mode,
        min_species_accept_confidence=species_conf,
    )

    # 支持的图片格式（包括RAW格式）
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
        ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".raf", ".pef", ".rw2"
    }

    # 获取所有图片文件
    image_files = [
        f for f in Path(input_folder).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"找到 {len(image_files)} 张图片")
    if crop_mode:
        print(f"裁剪模式已开启，裁剪图保存至: {_crop_dir}")
    print("=" * 50)

    total_crops = 0

    # 处理每张图片
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理 [{i}/{len(image_files)}]: {image_path.name}")

        try:
            result_image, results = detector.detect(str(image_path))

            # 保存标注结果
            base_name = image_path.stem
            output_filename = f"detected_{base_name}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            province = results.get("province")
            city     = results.get("city")
            print(f"  结果: {results['total_birds']} 只鸟, {results['total_eyes']} 个鸟眼")
            if province:
                print(f"  位置: {province}{city or ''}")
            print(f"  标注图保存: {output_path}")

            # 裁剪模式：读取原始图像（未标注）进行裁剪
            if crop_mode and results["total_birds"] > 0:
                raw_image = detector.load_image(str(image_path))
                saved = detector.crop_species(
                    image=raw_image,
                    birds=results["birds"],
                    output_dir=_crop_dir,
                    source_path=str(image_path),
                    counter=crop_counter,
                    margin_ratio=margin_ratio,
                    province=province,
                    city=city,
                )
                total_crops += len(saved)
                print(f"  裁剪图数量: {len(saved)} 张")

        except Exception as e:
            import traceback
            print(f"  错误: {e}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"处理完成！标注结果保存在: {output_folder}")
    if crop_mode:
        print(f"共生成裁剪图: {total_crops} 张，保存在: {_crop_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="鸟类目标检测 + 物种识别 + GPS 地理定位 + 分类裁剪",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 批量检测 + 标注
  python detect_bird_and_eye.py --input D:/photos --output D:/photos/detected

  # 批量检测 + 标注 + 裁剪（分类目录结构）
  python detect_bird_and_eye.py --input D:/photos --output D:/photos/detected --crop

  # 只裁剪，不保存标注图（加快速度）
  python detect_bird_and_eye.py --input D:/photos --output D:/photos/detected --crop --crop-only

  # 启用鸟眼检测
  python detect_bird_and_eye.py --input D:/photos --output D:/photos/detected --eye

  # 自定义裁剪边距（50%%）和保存目录
  python detect_bird_and_eye.py --input D:/photos --output D:/detected --crop --margin 0.5 --crop-dir D:/crops

  # 单张图片
  python detect_bird_and_eye.py --image D:/photos/bird.jpg --output D:/photos/detected --crop
""",
    )
    parser.add_argument(
        "--input", type=str,
        default=r"D:\BaiduNetdiskDownload\temp\test",
        help="输入图片文件夹",
    )
    parser.add_argument(
        "--output", type=str,
        default=r"D:\BaiduNetdiskDownload\temp\test\detected",
        help="输出结果文件夹（标注图）",
    )
    parser.add_argument(
        "--bird-model", type=str,
        default=str(_BIRDY_ROOT / "models" / "yolov8x-seg.pt"),
        help="鸟类检测模型路径",
    )
    parser.add_argument(
        "--eye-model", type=str,
        default=str(_BIRDY_ROOT / "models" / "birdeye.pt"),
        help="鸟眼检测模型路径",
    )
    parser.add_argument(
        "--species-model", type=str,
        default=_SPECIES_MODEL_PATH,
        help="物种识别模型路径（ResNet34）",
    )
    parser.add_argument(
        "--bird-info", type=str,
        default=_BIRD_INFO_PATH,
        help="物种信息 JSON 文件路径",
    )
    parser.add_argument(
        "--no-species", action="store_true",
        help="禁用物种识别（加快处理速度）",
    )
    parser.add_argument(
        "--eye", action="store_true",
        help="启用鸟眼检测（默认关闭）",
    )
    parser.add_argument(
        "--geo-mode", type=str, default="china",
        choices=["china", "auto", "none"],
        help="地理约束模式：china（默认，过滤非中国物种）| auto（有GPS按省，无GPS用china）| none（不约束）",
    )
    parser.add_argument(
        "--species-conf", type=float, default=0.5,
        help="未知种类阈值（默认 0.5），与 GUI「未知种类阈值」一致：低于此置信度的候选被过滤且顶一视为未知",
    )
    parser.add_argument(
        "--location", type=str, default=None,
        help="拍摄地点名称（如'杭州西湖'），自动转换为GPS坐标写入EXIF",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="单张图片路径（如果指定则只处理单张）",
    )
    # ── 裁剪相关参数 ──────────────────────────────────────────────
    parser.add_argument(
        "--crop", action="store_true",
        help="启用裁剪：按「目/科/属/种」目录结构保存，文件名含省市信息",
    )
    parser.add_argument(
        "--crop-only", action="store_true",
        help="仅生成裁剪图，不保存标注图（需同时指定 --crop）",
    )
    parser.add_argument(
        "--crop-dir", type=str, default="",
        help="裁剪图根目录（默认为 <output>/crops）",
    )
    parser.add_argument(
        "--margin", type=float, default=1.0,
        help="裁剪边距倍率（默认 1.0 = 四周各扩展 100%%框尺寸；0.5 = 50%%）",
    )

    args = parser.parse_args()
    enable_species = not args.no_species
    enable_eye     = args.eye
    do_crop        = args.crop or args.crop_only
    geo_mode       = args.geo_mode
    species_conf   = args.species_conf

    if args.image:
        # ── 单张图片处理 ──────────────────────────────────────────
        
        # 地理编码：地名 → GPS 坐标 → 写入 EXIF
        if args.location and _GEO_ENCODER_AVAILABLE:
            print(f"\n地理编码: '{args.location}' → GPS 坐标")
            coords = geocode_location(args.location)
            if coords:
                print(f"成功: ({coords[0]:.6f}, {coords[1]:.6f})")
                if write_gps_exif(args.image, coords[0], coords[1]):
                    print(f"✓ GPS EXIF 已写入单张图片")
                else:
                    print("✗ GPS EXIF 写入失败")
            else:
                print(f"✗ 地理编码失败: '{args.location}'")
        elif args.location and not _GEO_ENCODER_AVAILABLE:
            print("警告: --location 参数指定但 geo_encoder 模块不可用")
            print("请安装: pip install geopy piexif Pillow")
        
        detector = BirdAndEyeDetector(
            bird_model_path=args.bird_model,
            eye_model_path=args.eye_model,
            species_model_path=args.species_model,
            bird_info_path=args.bird_info,
            enable_species=enable_species,
            enable_eye=enable_eye,
            geo_mode=geo_mode,
            min_species_accept_confidence=species_conf,
        )

        print(f"处理单张图片: {args.image}")
        result_image, results = detector.detect(args.image)

        province = results.get("province")
        city     = results.get("city")

        os.makedirs(args.output, exist_ok=True)
        image_path_obj = Path(args.image)

        if not args.crop_only:
            output_filename = f"detected_{image_path_obj.stem}.jpg"
            output_path = os.path.join(args.output, output_filename)
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  标注图保存: {output_path}")

        if do_crop and results["total_birds"] > 0:
            _crop_dir = args.crop_dir if args.crop_dir else os.path.join(args.output, "crops")
            raw_image = detector.load_image(args.image)
            saved = detector.crop_species(
                image=raw_image,
                birds=results["birds"],
                output_dir=_crop_dir,
                source_path=args.image,
                margin_ratio=args.margin,
                province=province,
                city=city,
            )
            print(f"  裁剪图 {len(saved)} 张保存至: {_crop_dir}")

        print(f"\n检测完成!")
        print(f"  鸟数量:   {results['total_birds']}")
        print(f"  鸟眼数量: {results['total_eyes']}")
        if province:
            print(f"  拍摄位置: {province}{city or ''}")
    else:
        # ── 批量处理文件夹 ────────────────────────────────────────
        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            bird_model_path=args.bird_model,
            eye_model_path=args.eye_model,
            species_model_path=args.species_model,
            bird_info_path=args.bird_info,
            enable_species=enable_species,
            enable_eye=enable_eye,
            crop_mode=do_crop,
            crop_dir=args.crop_dir,
            margin_ratio=args.margin,
            geo_mode=geo_mode,
            species_conf=species_conf,
            location=getattr(args, 'location', None),
        )

