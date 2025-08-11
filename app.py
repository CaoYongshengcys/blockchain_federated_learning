import hashlib
import json
import time
from datetime import datetime
import random
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flask import Flask, jsonify, render_template_string

# --- 1. 区块链组件 ---

class Block:
    """定义区块链中的一个区块"""
    def __init__(self, index, timestamp, transactions, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """计算区块的哈希值"""
        block_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash
        }

class Blockchain:
    """管理整个区块链"""
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        """创建创世区块"""
        return Block(0, time.time(), [], "0")

    @property
    def latest_block(self):
        """获取最新的区块"""
        return self.chain[-1]

    def add_block(self):
        """添加新区块，并清空待处理交易"""
        if not self.pending_transactions:
            return None # 没有交易就不创建新区块
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions,
            previous_hash=self.latest_block.hash
        )
        self.pending_transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, transaction):
        """添加一笔新的交易"""
        self.pending_transactions.append(transaction)

    def to_dict_list(self):
        """将整个区块链转换为字典列表，方便JSON序列化"""
        return [block.to_dict() for block in self.chain]

# --- 2. 联邦学习组件 ---

class FederatedLearningAggregator:
    """中心聚合器，负责协调联邦学习过程"""
    def __init__(self):
        # 初始化一个全局模型。SGDClassifier支持在线学习，非常适合此场景。
        self.global_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
        # 为了演示，我们需要一个“虚拟”的初始数据集来fit模型
        # 实际应用中，模型可以无初始数据，或用公共数据集预训练
        X_initial = np.random.rand(10, 3) # 10个样本，3个特征
        y_initial = np.random.randint(0, 3, 10) # 3个类别 (0: idle, 1: charge, 2: discharge)
        self.scaler = StandardScaler().fit(X_initial)
        self.global_model.partial_fit(self.scaler.transform(X_initial), y_initial, classes=[0, 1, 2])
        self.history = [] # 记录全局模型的准确率历史

    def aggregate_models(self, local_models):
        """聚合本地模型（简单平均法）"""
        if not local_models:
            return

        # 提取所有本地模型的权重和截距
        weights = np.array([model.coef_ for model in local_models])
        intercepts = np.array([model.intercept_ for model in local_models])

        # 计算平均权重和截距
        avg_weight = np.mean(weights, axis=0)
        avg_intercept = np.mean(intercepts, axis=0)

        # 更新全局模型
        self.global_model.coef_ = avg_weight
        self.global_model.intercept_ = avg_intercept
        
        # 评估并记录新全局模型的性能
        self.evaluate_global_model()

    def evaluate_global_model(self):
        """在一个虚拟的测试集上评估全局模型性能"""
        # 创建一个平衡的虚拟测试集
        X_test = np.array([[0.8, 0.1, 10], [0.2, 0.8, 22], [0.5, 0.5, 16]]) # 高电量/低价, 低电量/高价, 中等
        y_test = np.array([2, 1, 0]) # 应该放电, 应该充电, 应该待机
        
        X_test_scaled = self.scaler.transform(X_test)
        accuracy = self.global_model.score(X_test_scaled, y_test)
        self.history.append(accuracy)
        print(f"FL Round Complete. Global Model Accuracy: {accuracy:.2f}")

# --- 3. 电动汽车 (EV) 和充电站模拟 ---

class ElectricVehicle:
    """定义一台电动汽车"""
    def __init__(self, ev_id, capacity_kwh=50.0):
        self.id = ev_id
        self.capacity = capacity_kwh
        self.soc = round(random.uniform(0.2, 0.8), 2) # 初始电量 (20%-80%)
        self.status = "idle" # "charging", "discharging", "idle"
        
        # 每个EV都有自己的本地模型和数据
        self.local_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        self.local_data_x = []
        self.local_data_y = []
        self.scaler = None

    def update_state(self, grid_price, time_of_day, global_model, scaler):
        """EV根据全局模型决定行为，并更新状态"""
        if self.scaler is None:
            self.scaler = scaler
            self.local_model.coef_ = global_model.coef_.copy()
            self.local_model.intercept_ = global_model.intercept_.copy()

        # 1. 使用全局模型进行预测
        features = np.array([[self.soc, grid_price, time_of_day]])
        features_scaled = self.scaler.transform(features)
        action = global_model.predict(features_scaled)[0] # 0: idle, 1: charge, 2: discharge

        # 2. 根据预测执行动作并更新SoC
        transaction = None
        kwh_amount = self.capacity * 0.05 # 每次充放5%的电量
        
        if action == 1 and self.soc < 0.95: # 充电
            self.status = "charging"
            self.soc = min(1.0, self.soc + 0.05)
            transaction = self.create_transaction(self.status, kwh_amount, grid_price)
        elif action == 2 and self.soc > 0.2: # 放电 (V2G)
            self.status = "discharging"
            self.soc = max(0.0, self.soc - 0.05)
            transaction = self.create_transaction(self.status, kwh_amount, grid_price)
        else: # 待机
            self.status = "idle"
        
        self.soc = round(self.soc, 2)

        # 3. 生成新的本地训练数据（模拟经验）
        # 简单规则：低价充电/高价放电是好经验
        if (grid_price < 0.3 and self.status == "charging") or \
           (grid_price > 0.7 and self.status == "discharging"):
            self.local_data_x.append(features[0])
            self.local_data_y.append(action)
            
        return transaction

    def train_local_model(self):
        """在本地数据上训练模型"""
        if len(self.local_data_x) < 5: # 数据太少不训练
            return None
        
        X = np.array(self.local_data_x)
        y = np.array(self.local_data_y)
        
        X_scaled = self.scaler.transform(X)
        
        # partial_fit允许模型在现有基础上继续学习
        self.local_model.partial_fit(X_scaled, y, classes=[0, 1, 2])
        return self.local_model

    def create_transaction(self, action, amount, price):
        """创建一笔交易记录"""
        return {
            "ev_id": self.id,
            "action": action,
            "amount_kwh": round(amount, 2),
            "price_per_kwh": price,
            "total_cost": round(amount * price, 2),
            "timestamp": time.time()
        }

    def to_dict(self):
        return {
            "id": self.id,
            "soc": self.soc,
            "status": self.status,
            "local_data_points": len(self.local_data_x)
        }

class ChargingStation:
    """主模拟器"""
    def __init__(self, num_evs=5):
        self.evs = [ElectricVehicle(ev_id=i) for i in range(num_evs)]
        self.blockchain = Blockchain()
        self.fl_aggregator = FederatedLearningAggregator()
        self.simulation_time = 0
        self.grid_price = 0.5
        self.grid_load = 0.5

    def simulation_step(self):
        """执行一个模拟步骤"""
        self.simulation_time += 1
        
        # 1. 更新电网状态（模拟价格波动）
        self.grid_price = round(0.5 + 0.4 * np.sin(self.simulation_time / 10), 2)
        self.grid_load = round(max(0.1, min(0.9, 0.5 + self.grid_price - 0.2)), 2)
        time_of_day = (self.simulation_time % 24) # 模拟一天中的时间

        # 2. 更新所有EV的状态
        for ev in self.evs:
            transaction = ev.update_state(
                self.grid_price, 
                time_of_day, 
                self.fl_aggregator.global_model,
                self.fl_aggregator.scaler
            )
            if transaction:
                self.blockchain.add_transaction(transaction)
        
        # 3. 将交易打包成新区块
        self.blockchain.add_block()

        # 4. 每10个步骤执行一轮联邦学习
        if self.simulation_time % 10 == 0:
            print(f"\n--- Starting Federated Learning Round {self.simulation_time // 10} ---")
            local_models = []
            # 随机选择一部分EV参与训练
            participants = random.sample(self.evs, k=max(1, len(self.evs) // 2))
            for ev in participants:
                model = ev.train_local_model()
                if model:
                    local_models.append(model)
            
            self.fl_aggregator.aggregate_models(local_models)
    
    def get_status(self):
        """获取整个系统的当前状态"""
        return {
            "simulation_time": self.simulation_time,
            "grid_price": self.grid_price,
            "grid_load": self.grid_load,
            "evs": [ev.to_dict() for ev in self.evs],
            "blockchain": self.blockchain.to_dict_list(),
            "fl_history": self.fl_aggregator.history
        }

# --- 4. Flask Web应用 ---

app = Flask(__name__)
station = ChargingStation(num_evs=8)

# HTML模板，内嵌CSS和JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>基于区块链与联邦学习的EV充放电管理系统</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f0f2f5; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: auto; }
        h1, h2 { text-align: center; color: #1a237e; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; transition: transform 0.2s; }
        .card:hover { transform: translateY(-5px); }
        .card h3 { margin-top: 0; color: #3f51b5; border-bottom: 2px solid #e8eaf6; padding-bottom: 10px; }
        .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: center; }
        .status-item { background: #e8eaf6; padding: 10px; border-radius: 6px; }
        .status-item .label { font-size: 0.9em; color: #5c6bc0; }
        .status-item .value { font-size: 1.5em; font-weight: bold; color: #3f51b5; }
        .ev-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .ev { border: 1px solid #ddd; border-radius: 6px; padding: 10px; text-align: center; }
        .ev-id { font-weight: bold; font-size: 1.1em; }
        .soc-bar { width: 100%; background-color: #e0e0e0; border-radius: 5px; overflow: hidden; height: 20px; margin: 5px 0; }
        .soc-fill { height: 100%; background-color: #4caf50; transition: width 0.5s; }
        .ev-status { font-size: 0.9em; font-style: italic; }
        .ev-status.charging { color: #4caf50; }
        .ev-status.discharging { color: #f44336; }
        .ev-status.idle { color: #757575; }
        .blockchain-container { max-height: 400px; overflow-y: auto; padding-right: 10px; }
        .block { border: 1px solid #ccc; border-radius: 6px; margin-bottom: 10px; padding: 10px; background: #fafafa; }
        .block-header { display: flex; justify-content: space-between; font-weight: bold; }
        .block-hash { font-family: monospace; font-size: 0.8em; color: #666; word-break: break-all; }
        .transaction { font-size: 0.9em; margin-top: 5px; padding-left: 10px; border-left: 2px solid #3f51b5; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>EV能源管理系统 (区块链 & 联邦学习)</h1>
        
        <div class="card">
            <h3>系统状态</h3>
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">模拟时间步</div>
                    <div class="value" id="sim-time">0</div>
                </div>
                <div class="status-item">
                    <div class="label">当前电价 (元/kWh)</div>
                    <div class="value" id="grid-price">0.00</div>
                </div>
                <div class="status-item">
                    <div class="label">电网负载</div>
                    <div class="value" id="grid-load">0%</div>
                </div>
                <div class="status-item">
                    <div class="label">区块数量</div>
                    <div class="value" id="block-count">0</div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>电动汽车 (EVs)</h3>
                <div id="ev-container" class="ev-container"></div>
            </div>
            <div class="card">
                <h3>联邦学习: 全局模型准确率</h3>
                <canvas id="fl-chart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>区块链浏览器</h3>
            <div id="blockchain-container" class="blockchain-container"></div>
        </div>
    </div>

    <script>
        let flChart;

        function initChart() {
            const ctx = document.getElementById('fl-chart').getContext('2d');
            flChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '全局模型准确率',
                        data: [],
                        borderColor: '#3f51b5',
                        backgroundColor: 'rgba(63, 81, 181, 0.1)',
                        fill: true,
                        tension: 0.2
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0
                        }
                    },
                    animation: {
                        duration: 500
                    }
                }
            });
        }

        function updateUI(data) {
            // 更新系统状态
            document.getElementById('sim-time').innerText = data.simulation_time;
            document.getElementById('grid-price').innerText = data.grid_price.toFixed(2);
            document.getElementById('grid-load').innerText = `${(data.grid_load * 100).toFixed(0)}%`;
            document.getElementById('block-count').innerText = data.blockchain.length;

            // 更新EVs
            const evContainer = document.getElementById('ev-container');
            evContainer.innerHTML = '';
            data.evs.forEach(ev => {
                const evDiv = document.createElement('div');
                evDiv.className = 'ev';
                evDiv.innerHTML = `
                    <div class="ev-id">EV #${ev.id}</div>
                    <div>SoC: ${(ev.soc * 100).toFixed(0)}%</div>
                    <div class="soc-bar">
                        <div class="soc-fill" style="width: ${ev.soc * 100}%;"></div>
                    </div>
                    <div class="ev-status ${ev.status}">${ev.status}</div>
                    <div style="font-size:0.8em; color:#999;">Data: ${ev.local_data_points}</div>
                `;
                evContainer.appendChild(evDiv);
            });

            // 更新区块链
            const blockchainContainer = document.getElementById('blockchain-container');
            blockchainContainer.innerHTML = '';
            [...data.blockchain].reverse().forEach(block => {
                const blockDiv = document.createElement('div');
                blockDiv.className = 'block';
                let transactionsHTML = block.transactions.map(tx => `
                    <div class="transaction">
                        EV #${tx.ev_id} -> ${tx.action}: ${tx.amount_kwh} kWh @ ${tx.price_per_kwh.toFixed(2)} 元
                    </div>
                `).join('');
                if (!transactionsHTML) transactionsHTML = '<div class="transaction"><i>创世区块 - 无交易</i></div>';
                
                blockDiv.innerHTML = `
                    <div class="block-header">
                        <span>区块 #${block.index}</span>
                        <span>${new Date(block.timestamp * 1000).toLocaleTimeString()}</span>
                    </div>
                    <div><small>Prev Hash: <span class="block-hash">${block.previous_hash.substring(0, 20)}...</span></small></div>
                    <div><small>Hash: <span class="block-hash">${block.hash.substring(0, 20)}...</span></small></div>
                    ${transactionsHTML}
                `;
                blockchainContainer.appendChild(blockDiv);
            });

            // 更新FL图表
            flChart.data.labels = data.fl_history.map((_, i) => `Round ${i + 1}`);
            flChart.data.datasets[0].data = data.fl_history;
            flChart.update();
        }

        async function fetchData() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                updateUI(data);
            } catch (error) {
                console.error("无法获取数据:", error);
            }
        }

        window.onload = () => {
            initChart();
            fetchData(); // 立即获取一次
            setInterval(fetchData, 2000); // 每2秒更新一次
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def get_status():
    """API端点，运行模拟并返回状态"""
    station.simulation_step()
    return jsonify(station.get_status())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)