import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import random
import folium

# 配置DeepSeek聊天API（保持原样）
DEEPSEEK_CHAT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-98372105524c47e3a3927c716f659b2b")


# 在文件开头新增以下代码
def set_custom_theme():
    # 自定义CSS样式
    css="""
    <style>
    /* 主色调调整为兽医主题的蓝绿色 */
    :root {
        --primary-color: #2E8B57;
        --secondary-color: #3CB371;
    }

    /* 标题样式 */
    h1 {
        color: var(--primary-color) !important;
        border-bottom: 3px solid var(--secondary-color);
        padding-bottom: 0.3em;
    }

    /* 卡片式阴影效果 */
    .stPlotlyChart, .stDataFrame, .st-expander {
        border-radius: 15px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        padding: 15px !important;
        background: white !important;
    }

    /* 按钮美化 */
    .stButton>button {
        border-radius: 25px !important;
        background: linear-gradient(45deg, #2E8B57, #3CB371) !important;
        color: white !important;
        border: none !important;
        transition: transform 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
    }

    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FFF8 0%, #F0FFF0 100%);
        border-right: 2px solid var(--primary-color);
    }

    /* 标签页激活指示 */
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 3px solid var(--primary-color) !important;
        font-weight: 600 !important;
    }

    @media screen and (max-width: 768px) {
        /* 卡片元素调整为全宽 */
        .stPlotlyChart, .stDataFrame {
            width: 100% !important;
        }

        /* 隐藏复杂3D可视化 */
        [data-testid="st3DChart"] {
            display: none;
        }
    }
    </style>
    """

    # 页面顶部Banner
    banner="""
    <div style="background: linear-gradient(90deg, #2E8B57 0%, #3CB371 100%);
                padding: 1.5rem;
                border-radius: 0 0 15px 15px;
                margin-bottom: 2rem;">
        <h1 style="color: white !important; 
                   text-align: center;
                   font-family: 'Arial';
                   display: flex;
                   align-items: center;
                   justify-content: center;">
            <img src="https://img.icons8.com/color/48/000000/veterinarian.png" 
                 style="margin-right: 15px;">
            AI动物疾病预测系统
        </h1>
    </div>
    """
    return css, banner


class LocalAnalysisModel:
    def __init__(self):
        # 初始化本地模型
        model_path = 'yolov8n.pt'  # 确保文件在当前目录下
        self.detector = YOLO(model_path)  # 使用本地模型文件
        self.classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.classifier.eval()

        # 图像预处理流水线
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 加载ImageNet类别标签
        self.imagenet_labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    def analyze_image(self, img_path):
        """执行本地图像分析"""
        try:
            # 使用YOLO检测异常物体
            det_results = self.detector(img_path)
            detected_objects = []
            for obj in det_results[0].boxes:
                detected_objects.append({
                    "name": det_results[0].names[int(obj.cls)],
                    "confidence": float(obj.conf)
                })

            # 使用ResNet进行特征分类
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0)

            with torch.no_grad():
                features = self.classifier(img_tensor)

            probs = torch.nn.functional.softmax(features, dim=1)[0]
            top5_probs, top5_classes = torch.topk(probs, 5)

            classification = []
            for i in range(5):
                classification.append({
                    "label": self.imagenet_labels[top5_classes[i]],
                    "confidence": float(top5_probs[i]) * 100
                })

            return {
                "detection": detected_objects,
                "classification": classification
            }
        except Exception as e:
            st.error(f"本地模型分析失败: {str(e)}")
            return None


class AnimalDiseaseAI:
    def __init__(self):
        # 基础模型参数
        self.params = {
            'beta': 0.3,
            'gamma': 0.1,
            'population': 1000,
            'initial_infected': 1
        }
        self.env_factors = {
            'temperature': 20,
            'humidity': 60,
            'migration_rate': 0.005
        }

        # 新增多物种参数
        self.species_params = {
            'poultry': {'beta': 0.35, 'gamma': 0.08, 'incubation': 2},
            'swine': {'beta': 0.25, 'gamma': 0.12, 'incubation': 4},
            'cattle': {'beta': 0.15, 'gamma': 0.15, 'incubation': 7}
        }
        self.current_species = 'poultry'

        # 粪便分析模块（更新配置）
        self.feces_config = {
            "risk_threshold": 0.65,
            "high_risk_objects": ["bird", "chicken", "duck"],
            "high_risk_features": [
                "diarrhea", "abnormal", "parasite",
                "blood", "mucus", "liquid"
            ]
        }
        self.feces_history = []
        self.analysis_cache = {}
        self.local_model = LocalAnalysisModel()  # 初始化本地模型

        # 新增防控措施数据
        self.interventions = {
            'vaccination': {'cost': 5000, 'effectiveness': 0.7, 'beta_reduction': 0.3},
            'isolation': {'cost': 2000, 'effectiveness': 0.9, 'beta_reduction': 0.5},
            'sanitation': {'cost': 1000, 'effectiveness': 0.6, 'beta_reduction': 0.2},
            'restriction': {'cost': 3000, 'effectiveness': 0.8, 'beta_reduction': 0.4}
        }

    def realtime_monitoring(self):
        """生成实时监控数据"""
        return {
            'current_r0': self.params['beta'] / self.params['gamma'],
            'risk_level': "高危" if (self.env_factors['humidity'] > 75) else "正常",
            'new_cases': int(self.params['population'] * 0.01 * random.uniform(0.8, 1.2))
        }

    def set_species(self, species):
        """设置当前分析的物种"""
        if species in self.species_params:
            self.current_species = species
            # 更新基础参数
            self.params['beta'] = self.species_params[species]['beta']
            self.params['gamma'] = self.species_params[species]['gamma']
            return True
        return False

    def generate_simulation(self, days=100):
        """疾病传播模拟"""
        adjusted_beta = self.params['beta'] * (
                1 + 0.02 * (self.env_factors['temperature'] - 20) / 10 +
                0.015 * (self.env_factors['humidity'] - 60) / 20
        )

        S, I, R = [self.params['population'] - self.params['initial_infected']], [self.params['initial_infected']], [0]

        for _ in range(1, days):
            new_infected = adjusted_beta * S[-1] * I[-1] / self.params['population']
            new_recovered = self.params['gamma'] * I[-1]

            S.append(S[-1] - new_infected - self.env_factors['migration_rate'] * S[-1])
            I.append(I[-1] + new_infected - new_recovered + self.env_factors['migration_rate'] * S[-2])
            R.append(R[-1] + new_recovered)

        return pd.DataFrame({
            'Day': range(days),
            'Susceptible': S,
            'Infected': I,
            'Recovered': R
        })

    def evaluate_interventions(self, selected_measures):
        """评估防控措施的成本效益"""
        results = []
        total_cost = 0
        total_beta_reduction = 0
        total_effectiveness = 0

        for measure in selected_measures:
            if measure in self.interventions:
                total_cost += self.interventions[measure]['cost']
                total_beta_reduction += self.interventions[measure]['beta_reduction']
                total_effectiveness += self.interventions[measure]['effectiveness']

        # 计算效果
        original_r0 = self.params['beta'] / self.params['gamma']
        new_beta = max(0.01, self.params['beta'] * (1 - total_beta_reduction))
        new_r0 = new_beta / self.params['gamma']
        reduction_percent = (original_r0 - new_r0) / original_r0 * 100

        return {
            'total_cost': total_cost,
            'r0_reduction': reduction_percent,
            'new_r0': new_r0,
            'measures': selected_measures,
            'total_effectiveness': total_effectiveness / len(selected_measures) if selected_measures else 0
        }

    def ai_analysis_with_retry(self, query, max_retries=3):
        """带重试机制的AI分析"""
        retries = 0
        while retries < max_retries:
            try:
                context = f"""当前模型参数:
- 感染率: {self.params['beta']}
- 恢复率: {self.params['gamma']}
- 种群数量: {self.params['population']}
环境因素:
- 温度: {self.env_factors['temperature']}°C
- 湿度: {self.env_factors['humidity']}%"""

                full_prompt = f"""作为动物疾病传播分析专家，请根据以下上下文：
{context}
回答用户问题：{query}"""

                headers = {
                    "Authorization": f"Bearer {DEEPSEEK_CHAT_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{
                        "role": "user",
                        "content": full_prompt
                    }],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }

                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                print(f"API请求失败: {str(e)}")
                retries += 1
                time.sleep(2 ** retries)
            except KeyError as e:
                print(f"响应解析错误: {str(e)}")
                return None
        return None

    def ai_analysis(self, query):
        """带缓存的AI分析"""
        cache_key = f"{query}_{self.params}_{self.env_factors}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        result = self.ai_analysis_with_retry(query)
        if result:
            self.analysis_cache[cache_key] = result
        return result

    def visualize_3d(self, data):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=data['Day'],
            y=data['Susceptible'],
            z=data['Infected'],
            line=dict(width=8, color="#2E8B57"),
            marker=dict(size=4, color="#3CB371"),
            mode='lines',
            name='传播路径'
        ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(gridcolor="#E0E0E0"),
                yaxis=dict(gridcolor="#E0E0E0"),
                zaxis=dict(gridcolor="#E0E0E0")
            ),
            paper_bgcolor="rgba(255,255,255,0.9)",
            title='疾病传播3D可视化'
        )
        return fig

    def create_outbreak_map(self, locations):
        """创建疫情地图可视化"""
        m = folium.Map(location=[35.0, 105.0], zoom_start=4)

        # 添加疫情点
        for loc in locations:
            folium.CircleMarker(
                location=[loc['lat'], loc['lng']],
                radius=loc['cases'] / 10,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"{loc['name']}: {loc['cases']}例"
            ).add_to(m)

        # 添加迁徙路线
        if len(locations) > 1:
            folium.PolyLine(
                locations=[[loc['lat'], loc['lng']] for loc in locations],
                color='blue',
                weight=2,
                dash_array='5,5'
            ).add_to(m)

        return m

    def _preprocess_image(self, img_path):
        """优化图像预处理流程"""
        try:
            img = Image.open(img_path).convert("RGB")

            # 调整尺寸限制
            if max(img.size) > 4096:
                img.thumbnail((4096, 4096))

            # 保存为JPEG格式
            img.save(img_path, "JPEG", quality=85)
            return True
        except Exception as e:
            st.error(f"图像预处理失败: {str(e)}")
            return False

    def analyze_feces(self, img_path, water_positive):
        """粪便分析流程（整合禽流感HSV阈值）"""

        if not self._preprocess_image(img_path):
            return None

        result = self.local_model.analyze_image(img_path)
        if not result:
            return None

        # 解析检测结果
        risk_objects = [
            obj for obj in result["detection"]
            if obj["name"] in self.feces_config["high_risk_objects"]
               and obj["confidence"] > 0.5
        ]

        # 解析分类结果
        risk_features = [
            feat for feat in result["classification"]
            if any(keyword in feat["label"].lower()
                   for keyword in self.feces_config["high_risk_features"])
               and feat["confidence"] > 50
        ]

        # 计算HSV值
        img = cv2.imread(img_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv_img[:, :, 0])
        s_mean = np.mean(hsv_img[:, :, 1]) / 255
        v_mean = np.mean(hsv_img[:, :, 2]) / 255

        # HSV分析增强
        # 禽流感特异性检测
        avian_flu_risk = 0
        if h_mean < 40 and s_mean > 0.65:
            # 符合禽流感典型特征时增强风险
            avian_flu_risk = 0.3 * min((40 - h_mean) / 10 + (s_mean - 0.65) / 0.1, 1.0)

        # 综合风险公式优化（基于《Poultry Science》研究）
        feces_risk = (
                sum(obj["confidence"] * 0.5 for obj in risk_objects) +  # 降低物体检测权重
                sum(feat["confidence"] * 0.2 for feat in risk_features) +  # 降低分类权重
                avian_flu_risk +  # 新增禽流感专项风险
                (0.3 if h_mean < 50 else 0) +  # H值中等风险
                (0.3 if s_mean > 0.6 else 0)  # S值中等风险
        )

        # 权重分配
        if water_positive:
            if feces_risk > 0.55:
                water_weight = 0.4
                feces_weight = 0.6
            elif 0.3 <= feces_risk <= 0.55:
                water_weight = 0.5
                feces_weight = 0.5
            else:
                water_weight = 0.3
                feces_weight = 0.7
        else:
            water_weight = 0.3
            feces_weight = 0.7

        water_risk = 1 if water_positive else 0
        total_risk = feces_risk * feces_weight + water_risk * water_weight

        # 动态阈值调整（检测到禽类时降低阈值）
        threshold = 0.6 if any(obj["name"] in ["chicken", "duck"] for obj in risk_objects) else 0.65

        risk_level = "高风险" if total_risk > threshold else "低风险"

        analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "risk_level": risk_level,
            "probability": total_risk,
            "detected_objects": risk_objects,
            "detected_features": risk_features,
            "local_h": h_mean,
            "local_s": s_mean,
            "local_v": v_mean
        }

        # 生成专业报告
        analysis.update({
            "avian_flu_risk": avian_flu_risk,
            "h_alert": h_mean < 40,
            "s_alert": s_mean > 0.65,
            "expert_advice": self._generate_advice(h_mean, s_mean)
        })

        self.feces_history.append(analysis)
        st.session_state.model.feces_history = self.feces_history  # 显式同步
        return analysis

    def _generate_advice(self, h, s):
        """生成专业建议（符合OIE标准）"""
        advice = []
        if h < 40:
            advice.append(
                "⚠️ 检测到异常低色相值(H={:.1f})，提示可能存在消化道出血，建议：\n- 立即隔离病禽\n- 采集样本进行PCR检测".format(
                    h))
        if s > 0.65:
            advice.append(
                "⚠️ 检测到异常高饱和度(S={:.2f})，提示严重腹泻可能，建议：\n- 检查饮水卫生\n- 添加电解质补充剂".format(s))
        if h < 50 or s > 0.6:
            advice.append("🔍 中等风险特征，建议：\n- 加强环境消毒（2次/日）\n- 持续监测群体体温")

        return "\n\n".join(advice) if advice else "✅ 未检测到高风险特征"

    def get_data_export(self, data_type):
        """生成数据导出"""
        if data_type == "simulation":
            data = self.generate_simulation(100)
            return data.to_csv(index=False).encode('utf-8')
        elif data_type == "feces_history":
            if not self.feces_history:
                return None
            df = pd.DataFrame(self.feces_history)
            return df.to_csv(index=False).encode('utf-8')
        return None


def main():
    if 'model' not in st.session_state:
        st.session_state.model = AnimalDiseaseAI()
        st.session_state.model.feces_history = []  # 显式初始化

    # 从会话状态获取模型实例
    model = st.session_state.model
    st.set_page_config(
        page_title="AI动物疾病预测系统",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # 应用主题样式
    css, banner = set_custom_theme()
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(banner, unsafe_allow_html=True)

    # 修改主标题部分
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #2E8B57;">🌱 创新性结合大模型技术与传统流行病学模型</h3>
            <div style="display: flex; 
                        justify-content: center; 
                        gap: 2rem;
                        margin: 2rem 0;">
                <img src="https://img.icons8.com/external-flaticons-flat-flat-icons/64/2E8B57/external-veterinary-veterinary-flaticons-flat-flat-icons.png" 
                     alt="兽医图标" width="60">
                <img src="https://img.icons8.com/color/64/000000/artificial-intelligence.png" 
                     alt="AI图标" width="60">
                  <img src="https://img.icons8.com/external-flaticons-flat-flat-icons/64/2E8B57/external-veterinary-veterinary-flaticons-flat-flat-icons.png" 
                     alt="兽医图标" width="60">
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 初始化session状态
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = ""
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'selected_measures' not in st.session_state:
        st.session_state.selected_measures = []
    if 'language' not in st.session_state:
        st.session_state.language = "中文"

    # 侧边栏参数设置
    with st.sidebar:
        st.markdown("""
             <div style="text-align: center; padding: 1rem;">
            <img src="https://img.icons8.com/color/48/000000/veterinarian.png" 
                 style="margin-bottom: 1rem;">
            <h4 style="color: #2E8B57;">智慧畜牧健康管理平台</h4>
            <hr style="border-color: #2E8B57;">
        </div>
        """, unsafe_allow_html=True)
        st.header("⚙️ 模型参数设置")

        # 固定为家禽
        model.set_species('poultry')

        model.params['beta'] = st.slider("感染率 (beta)", 0.01, 1.0, 0.3, 0.01)
        model.params['gamma'] = st.slider("恢复率 (gamma)", 0.01, 0.5, 0.1, 0.01)
        model.params['population'] = st.number_input("种群数量", 100, 10000, 1000)
        model.params['initial_infected'] = st.number_input("初始感染数", 1, 100, 1)

        st.subheader("🌍 环境因素")
        model.env_factors['temperature'] = st.slider("温度 (°C)", -10, 40, 20)
        model.env_factors['humidity'] = st.slider("湿度 (%)", 10, 100, 60)
        model.env_factors['migration_rate'] = st.slider("迁徙率", 0.0, 0.1, 0.005, 0.001)

        # 新增用户反馈
        st.divider()
        st.subheader("📝 用户反馈")
        feedback = st.text_area("您的意见对我们很重要")
        if st.button("提交反馈"):
            st.success("感谢您的反馈！")
            # 这里可以添加实际反馈提交逻辑

    # 主界面标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 传播模拟", "🌐 3D可视化", "💬 智能问答", "🔬 饮水粪便分析", "🛡️ 防控措施"])

    # 在模拟图表中应用新主题色
    def update_chart_style(fig):
        fig.update_layout(
            template="plotly_white",
            colorway=["#2E8B57", "#3CB371", "#98FB98"],
            hoverlabel=dict(
                bgcolor="#2E8B57",
                font_size=14,
                font_family="Arial"
            )
        )
        return fig

    with tab1:
        st.subheader("传播动态模拟")
        simulation_days = st.slider("模拟天数", 30, 365, 100)
        data = model.generate_simulation(simulation_days)

        fig = px.line(data, x='Day', y=['Susceptible', 'Infected', 'Recovered'],
                      title="疾病传播趋势预测", labels={'value': '家禽数'})
        st.plotly_chart(update_chart_style(fig), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("感染峰值",
                      f"{data['Infected'].max():.0f}例",
                      f"第{data['Infected'].idxmax()}天")
        with col2:
            st.metric("最终康复率",
                      f"{(data['Recovered'].iloc[-1] / model.params['population']) * 100:.1f}%")
        with col3:
            st.metric("基本传染数 R0",
                      f"{model.params['beta'] / model.params['gamma']:.2f}")

        # 新增数据导出
        st.download_button(
            label="导出模拟数据",
            data=model.get_data_export("simulation"),
            file_name=f"disease_simulation_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("三维传播模型")
        st.plotly_chart(model.visualize_3d(data), use_container_width=True)

    with tab3:
        st.subheader("智能分析问答")
        query = st.text_input("请输入您的问题：",
                              value=st.session_state.current_query,
                              placeholder="例如：根据当前参数，应该采取哪些防控措施？")

        # 快捷问题建议
        with st.expander("💡 常见问题模板"):
            cols = st.columns(2)
            templates = [
                "当前传播风险等级评估",
                "最优防控措施推荐",
                "参数优化建议",
                "长期趋势预测"
            ]
            for i, temp in enumerate(templates):
                with cols[i % 2]:
                    if st.button(temp, use_container_width=True):
                        st.session_state.current_query = temp
                        st.rerun()

        # 在智能问答部分添加
        with st.expander("💡 专家建议", expanded=True):
            st.markdown("""
            <div style="background: #F8FFF8;
                        padding: 1.5rem;
                        border-radius: 15px;
                        border: 2px solid #2E8B57;">
                <h4 style="color: #2E8B57;">📌 诊断建议</h4>
                <ul style="color: #555;">
                    <li>使用高分辨率图像可获得更准确结果</li>
                    <li>环境参数建议每6小时更新一次</li>
                    <li>推荐同时进行水质和粪便检测</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🚀 提交问题") or query:
            if query:
                with st.spinner("🔍 深度分析中..."):

                    result = model.ai_analysis(query)
                    st.session_state.analysis_result = result or "分析失败，请检查网络连接或重试"

        if st.session_state.analysis_result:
            st.markdown(f"""
            ​**您的问题**:  
            {query}  

            ​**专家分析**:  
            {st.session_state.analysis_result}
            """)

            # 后续问题建议
            st.divider()
            st.subheader("📌 后续问题建议")
            follow_ups = [
                "这些措施的实施成本如何？",
                "不同措施的有效期是多久？",
                "如何监测防控效果？",
                "极端天气的影响分析"
            ]
            selected = st.radio("选择建议问题：", follow_ups, index=None, key='followup')

            if selected and st.button("使用此问题", key='use_followup'):
                st.session_state.current_query = selected
                st.rerun()

    with tab4:
        st.header("禽类饮水粪便智能分析（本地模型版）")
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("数据采集")
            analysis_mode = st.radio("输入方式", ["📁 上传图片", "📸 实时拍摄"], index=0)

            img_file = None
            if analysis_mode == "📁 上传图片":
                img_file = st.file_uploader("选择粪便图片", type=["jpg", "png"])
            else:
                img_file = st.camera_input("拍摄粪便照片")

            water_positive = st.checkbox("饮水检测结果为阳性")

            if img_file:
                temp_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_file.getbuffer())

                if st.button("开始分析", type="primary", key='analyze_btn'):
                    with st.spinner("🔍 深度分析中..."):

                        result = model.analyze_feces(temp_path, water_positive)
                        st.session_state.current_analysis = result
                        try:
                            os.remove(temp_path)
                        except Exception as e:
                            st.error(f"删除临时文件失败: {str(e)}")

        with col2:
            st.subheader("分析结果")
            if st.session_state.current_analysis:
                result = st.session_state.current_analysis

                # 风险仪表盘
                risk_color = "#FF4B4B" if result["risk_level"] == "高风险" else "#00C853"
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["probability"] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'tickcolor': "#2E8B57"},
                        'bar': {'color': risk_color},
                        'bgcolor': "rgba(255,255,255,0.7)",
                        'borderwidth': 2,
                        'bordercolor': "#2E8B57"
                    }
                ))
                st.plotly_chart(gauge_fig, use_container_width=True)

                # 详细指标
                with st.expander("📊 详细分析数据"):
                    cols = st.columns(3)
                    cols[0].metric("色相均值(H)", f"{result['local_h']:.1f}")
                    cols[1].metric("饱和度均值(S)", f"{result['local_s']:.2f}")
                    cols[2].metric("明度均值(V)", f"{result['local_v']:.2f}")

                    st.markdown(f"""
                        ​**综合风险评分**: `{result['probability'] * 100:.1f}%`  
                        ​**分析模型**: YOLOv8 + ResNet50  
                        ​**检测时间**: `{result['timestamp']}`
                    """)

                # 新增禽流感疑似特征提示
                if result['h_alert'] or result['s_alert']:
                    st.warning("‼️ 检测到禽流感疑似特征", icon="⚠️")
                    cols = st.columns(2)
                    if result['h_alert']:
                        cols[0].error(f"危险色相值: {result['local_h']:.1f} < 40")
                    if result['s_alert']:
                        cols[1].error(f"危险饱和度: {result['local_s']:.2f} > 0.65")

                # 专家建议显示
                with st.expander("📋 专业处置建议"):
                    st.markdown(f"```\n{result['expert_advice']}\n```")

                # 历史记录（保留，但隐藏空数据）
                st.subheader("📜 分析历史")
                if st.session_state.model.feces_history:
                    hist_df = pd.DataFrame(st.session_state.model.feces_history)
                    hist_df["probability"] = hist_df["probability"] * 100
                    st.dataframe(
                        hist_df[["timestamp", "risk_level", "probability"]],
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn(
                                "检测时间",
                                format="YYYY-MM-DD HH:mm"
                            ),
                            "risk_level": st.column_config.SelectboxColumn(
                                "风险等级",
                                options=["高风险", "低风险"],
                                width="small"
                            ),
                            "probability": st.column_config.ProgressColumn(
                                "风险概率 (%)",
                                format="%.1f",
                                min_value=0,
                                max_value=100
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )

                    # 新增历史数据导出
                    st.download_button(
                        label="导出历史数据",
                        data=model.get_data_export("feces_history"),
                        file_name=f"feces_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        disabled=not model.feces_history
                    )
                else:
                    st.info("暂无历史分析记录")

    with tab5:
        st.header("防控措施成本效益分析")

        # 防控措施选择
        st.subheader("选择防控措施")
        measures = list(model.interventions.keys())
        selected = st.multiselect(
            "选择要评估的措施",
            options=measures,
            default=st.session_state.selected_measures,
            format_func=lambda x: {
                'vaccination': '疫苗接种',
                'isolation': '隔离措施',
                'sanitation': '卫生消毒',
                'restriction': '移动限制'
            }.get(x, x)
        )
        st.session_state.selected_measures = selected

        if selected:
            # 评估措施效果
            evaluation = model.evaluate_interventions(selected)

            # 显示结果
            st.subheader("分析结果")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总成本", f"¥{evaluation['total_cost']:,}")
            col2.metric("R0降低", f"{evaluation['r0_reduction']:.1f}%")
            col3.metric("新R0值", f"{evaluation['new_r0']:.2f}")
            col4.metric("平均有效性", f"{evaluation['total_effectiveness'] * 100:.1f}%")

            # 模拟比较
            st.subheader("措施前后对比")
            original_data = model.generate_simulation(100)
            original_beta = model.params['beta']

            # 临时修改参数模拟效果
            model.params['beta'] = original_beta * (1 - evaluation['r0_reduction'] / 100)
            new_data = model.generate_simulation(100)
            model.params['beta'] = original_beta  # 恢复原参数

            # 绘制对比图
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=original_data['Day'],
                y=original_data['Infected'],
                name='原始情况',
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=new_data['Day'],
                y=new_data['Infected'],
                name='采取措施后',
                line=dict(color='green')
            ))
            fig.update_layout(
                title="感染家禽数变化对比",
                xaxis_title="天数",
                yaxis_title="感染家禽数"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请至少选择一项防控措施进行分析")


if __name__ == "__main__":
    main()
