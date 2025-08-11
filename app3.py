import hashlib
import json
import time
from datetime import datetime
import random
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from flask import Flask, jsonify, render_template_string

# --- 1. 区块链组件 (修复了 to_dict 中的哈希值问题) ---
class Block:
    def __init__(self, index, timestamp, transactions, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
    def calculate_hash(self):
        # 使用确定的键来计算哈希，避免因 to_dict 变化而导致哈希不一致
        block_string = json.dumps({"index": self.index, "timestamp": self.timestamp, "transactions": self.transactions, "previous_hash": self.previous_hash}, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    def to_dict(self):
        # 确保哈希值也被包含在字典中
        return {"index": self.index, "timestamp": self.timestamp, "transactions": self.transactions, "previous_hash": self.previous_hash, "hash": self.hash}

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
    def create_genesis_block(self):
        return Block(0, time.time(), [], "0")
    @property
    def latest_block(self):
        return self.chain[-1]
    def add_block(self, add_log_callback):
        if not self.pending_transactions:
            return None
        add_log_callback(f"[Blockchain] 开始打包 {len(self.pending_transactions)} 条交易到新区块...", "blockchain")
        block = Block(index=len(self.chain), timestamp=time.time(), transactions=self.pending_transactions, previous_hash=self.latest_block.hash)
        self.pending_transactions = []
        self.chain.append(block)
        add_log_callback(f"[Blockchain] 成功创建区块 #{block.index}！", "blockchain")
        return block
    def add_transaction(self, transaction, add_log_callback):
        add_log_callback(f"[EV] EV #{transaction['ev_id']} 创建了新的交易 ({transaction['action']})", "ev")
        self.pending_transactions.append(transaction)
    def to_dict_list(self):
        return [block.to_dict() for block in self.chain]

# --- 2. 联邦学习组件 (修改了评估函数) ---
class FederatedLearningAggregator:
    def __init__(self):
        self.global_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
        X_initial = np.random.rand(10, 3)
        y_initial = np.random.randint(0, 3, 10)
        self.scaler = StandardScaler().fit(X_initial)
        self.global_model.partial_fit(self.scaler.transform(X_initial), y_initial, classes=[0, 1, 2])
        self.history = []
        
    def aggregate_models(self, local_models, participants_ids, X_test, y_test, add_log_callback):
        if not local_models:
            add_log_callback("[FL] 没有有效的本地模型可供聚合。", "fl")
            return
        add_log_callback(f"[FL] 正在聚合 {len(local_models)} 个本地模型...", "fl")
        weights = np.array([model.coef_ for model in local_models])
        intercepts = np.array([model.intercept_ for model in local_models])
        avg_weight = np.mean(weights, axis=0)
        avg_intercept = np.mean(intercepts, axis=0)
        self.global_model.coef_ = avg_weight
        self.global_model.intercept_ = avg_intercept
        add_log_callback("[FL] 聚合完成，全局模型已更新！", "fl")
        self.evaluate_local_models(local_models, participants_ids, X_test, y_test, add_log_callback)
        return {"local_weights": [m.coef_[0].tolist() for m in local_models], "global_weights": avg_weight[0].tolist()}

    def evaluate_local_models(self, local_models, participants_ids, X_test, y_test, add_log_callback):
        # 评估各个局部模型的准确率
        X_test_scaled = self.scaler.transform(X_test)
        local_accuracies = []
        for i, model in enumerate(local_models):
            accuracy = model.score(X_test_scaled, y_test)
            local_accuracies.append(accuracy)
            add_log_callback(f"[FL] EV #{participants_ids[i]} 本地模型准确率: {accuracy:.3f}", "fl")
        
        # 计算平均局部模型准确率并记录历史
        avg_accuracy = np.mean(local_accuracies)
        self.history.append(avg_accuracy)
        add_log_callback(f"[FL] 平均局部模型准确率: {avg_accuracy:.3f}", "fl")

# --- 3. 电动汽车 (EV) 和充电站模拟 (重大修改) ---
class ElectricVehicle:
    # ... (内部逻辑无变化) ...
    def __init__(self, ev_id, capacity_kwh=50.0):
        self.id = ev_id; self.capacity = capacity_kwh; self.soc = round(random.uniform(0.2, 0.8), 2); self.status = "idle"; self.is_training = False; self.local_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3); self.local_data_x = []; self.local_data_y = []; self.scaler = None
    def update_state(self, grid_price, time_of_day, global_model, scaler):
        if self.scaler is None: self.scaler = scaler; self.local_model.coef_ = global_model.coef_.copy(); self.local_model.intercept_ = global_model.intercept_.copy()
        features = np.array([[self.soc, grid_price, time_of_day]]); features_scaled = self.scaler.transform(features); action = global_model.predict(features_scaled)[0]
        transaction = None; kwh_amount = self.capacity * 0.05
        if action == 1 and self.soc < 0.95: self.status = "charging"; self.soc = min(1.0, self.soc + 0.05); transaction = self.create_transaction(self.status, kwh_amount, grid_price)
        elif action == 2 and self.soc > 0.2: self.status = "discharging"; self.soc = max(0.0, self.soc - 0.05); transaction = self.create_transaction(self.status, kwh_amount, grid_price)
        else: self.status = "idle"
        self.soc = round(self.soc, 2)
        if (grid_price < 0.3 and self.status == "charging") or (grid_price > 0.7 and self.status == "discharging"): self.local_data_x.append(features[0]); self.local_data_y.append(action)
        return transaction
    def train_local_model(self, add_log_callback):
        if len(self.local_data_x) < 5: return None
        add_log_callback(f"[FL] EV #{self.id} 正在训练本地模型...", "fl")
        X, y = np.array(self.local_data_x), np.array(self.local_data_y); X_scaled = self.scaler.transform(X)
        self.local_model.partial_fit(X_scaled, y, classes=[0, 1, 2])
        add_log_callback(f"[FL] 收到来自 EV #{self.id} 的本地模型更新。", "fl")
        return self.local_model
    def create_transaction(self, action, amount, price):
        return {"ev_id": self.id, "action": action, "amount_kwh": round(amount, 2), "price_per_kwh": price, "total_cost": round(amount * price, 2), "timestamp": datetime.now().isoformat()}
    def to_dict(self):
        return {"id": self.id, "soc": self.soc, "status": self.status, "local_data_points": len(self.local_data_x), "is_training": self.is_training}

class ChargingStation:
    def __init__(self, num_evs=8, fl_interval=10):
        self.evs = [ElectricVehicle(ev_id=i) for i in range(num_evs)]
        self.blockchain = Blockchain()
        self.fl_aggregator = FederatedLearningAggregator()
        self.simulation_time = 0; self.grid_price = 0.5; self.grid_load = 0.5
        self.fl_interval = fl_interval
        self.fl_status_text = "等待下一轮"; self.fl_participants = []
        self.logs = deque(maxlen=20)
        self.fl_round_details = {}
        
        # 新增：创建更健壮的全局测试集
        self.create_global_test_set(50)

    def create_global_test_set(self, n_samples):
        self.add_log("[System] 创建全局测试数据集...", "system")
        soc = np.random.uniform(0.1, 0.9, n_samples)
        price = np.random.uniform(0.1, 0.9, n_samples)
        time_of_day = np.random.randint(0, 24, n_samples)
        self.global_X_test = np.column_stack([soc, price, time_of_day])
        
        # 定义简单的逻辑来确定“正确”的行为
        # 低电量或低价格 -> 充电 (1)
        # 高电量和高价格 -> 放电 (2)
        # 其他 -> 待机 (0)
        self.global_y_test = np.zeros(n_samples, dtype=int)
        self.global_y_test[(soc < 0.3) | (price < 0.3)] = 1
        self.global_y_test[(soc > 0.7) & (price > 0.7)] = 2

    def add_log(self, message, category="system"):
        self.logs.appendleft({"time": datetime.now().strftime("%H:%M:%S"), "message": message, "category": category})

    def simulation_step(self):
        # 核心修复：调整逻辑顺序
        # 1. 在步骤开始时，处理上一轮的待定交易
        self.blockchain.add_block(self.add_log)

        # 2. 正常进行模拟
        self.simulation_time += 1
        self.grid_price = round(0.5 + 0.4 * np.sin(self.simulation_time / 10), 2)
        self.grid_load = round(max(0.1, min(0.9, 0.5 + self.grid_price - 0.2)), 2)
        time_of_day = (self.simulation_time % 24)

        for ev in self.evs: ev.is_training = False
        self.fl_participants = []
        self.fl_round_details = {}

        countdown = self.fl_interval - (self.simulation_time % self.fl_interval)
        if countdown == 1: self.fl_status_text = "准备训练..."
        elif self.simulation_time > 0 and self.simulation_time % self.fl_interval == 0:
            self.add_log(f"[FL] 开始第 {self.simulation_time // self.fl_interval} 轮联邦学习...", "fl")
            self.fl_status_text = "正在聚合模型..."
            
            participants = random.sample(self.evs, k=max(1, len(self.evs) // 2))
            self.fl_participants = [p.id for p in participants]
            local_models = []
            
            for ev in participants:
                ev.is_training = True
                model = ev.train_local_model(self.add_log)
                if model: local_models.append(model)
            
            # 核心修复：将测试集传入聚合器
            aggregation_result = self.fl_aggregator.aggregate_models(local_models, self.fl_participants, self.global_X_test, self.global_y_test, self.add_log)
            if aggregation_result:
                self.fl_round_details = {"participants": self.fl_participants, "aggregation": aggregation_result}
        else: self.fl_status_text = f"等待中 ({countdown}步后)"

        # 3. 生成本轮的新交易，它们将留在Mempool中，直到下一个步骤开始
        for ev in self.evs:
            transaction = ev.update_state(self.grid_price, time_of_day, self.fl_aggregator.global_model, self.fl_aggregator.scaler)
            if transaction: self.blockchain.add_transaction(transaction, self.add_log)
    
    def get_status(self):
        return {
            "simulation_time": self.simulation_time, "grid_price": self.grid_price, "grid_load": self.grid_load,
            "evs": [ev.to_dict() for ev in self.evs],
            "blockchain": self.blockchain.to_dict_list(),
            "pending_transactions": self.blockchain.pending_transactions,
            "fl_history": self.fl_aggregator.history, "fl_status": self.fl_status_text, "fl_participants": self.fl_participants,
            "logs": list(self.logs),
            "fl_round_details": self.fl_round_details
        }

# --- 4. Flask Web应用 (前端HTML/JS无需修改，但为完整性一并提供) ---
app = Flask(__name__)
station = ChargingStation()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EV能源管理系统 (区块链 & 联邦学习)</title>
    <style>
        :root { --primary-color: #3f51b5; --accent-color: #ff9800; --bg-light: #f0f2f5; --card-bg: white; --text-dark: #333; --text-light: #666; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: var(--bg-light); color: var(--text-dark); margin: 0; padding: 20px; }
        .container { max-width: 1600px; margin: auto; }
        h1 { text-align: center; color: #1a237e; }
        .main-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .card { background: var(--card-bg); border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; transition: all 0.3s; }
        .card h3 { margin-top: 0; color: var(--primary-color); border-bottom: 2px solid #e8eaf6; padding-bottom: 10px; }
        .status-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center; margin-bottom: 20px;}
        .status-item { background: #e8eaf6; padding: 10px; border-radius: 6px; }
        .status-item .label { font-size: 0.9em; color: #5c6bc0; }
        .status-item .value { font-size: 1.5em; font-weight: bold; color: var(--primary-color); }
        .ev-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; }
        .ev { border: 2px solid #ddd; border-radius: 6px; padding: 10px; text-align: center; transition: all 0.3s; }
        .ev.training { border-color: var(--accent-color); box-shadow: 0 0 10px rgba(255, 152, 0, 0.5); }
        .ev-id { font-weight: bold; } .soc-bar { width: 100%; background-color: #e0e0e0; border-radius: 5px; overflow: hidden; height: 20px; margin: 5px 0; } .soc-fill { height: 100%; background-color: #4caf50; transition: width 0.5s; } .ev-status { font-size: 0.9em; font-style: italic; } .ev-status.charging { color: #4caf50; } .ev-status.discharging { color: #f44336; } .ev-status.idle { color: #757575; }
        .log-container { height: 300px; overflow-y: auto; background: #2d2d2d; color: #f0f0f0; font-family: monospace; font-size: 0.9em; padding: 15px; border-radius: 6px; }
        .log-item { margin-bottom: 5px; white-space: pre-wrap; }
        .log-item .time { color: #999; } .log-item .cat-ev { color: #81d4fa; } .log-item .cat-blockchain { color: #a5d6a7; } .log-item .cat-fl { color: #ffd54f; } .log-item .cat-system { color: #cfcfcf; }
        .blockchain-container { height: 300px; overflow-y: auto; }
        .block { border: 1px solid #ccc; border-radius: 6px; margin-bottom: 10px; padding: 10px; background: #fafafa; transition: background-color 0.5s; }
        .block.new-block { background-color: #c8e6c9; animation: flash 1.5s ease-out; }
        @keyframes flash { 0% { background-color: #a5d6a7; } 100% { background-color: #fafafa; } }
        .block-header { display: flex; justify-content: space-between; font-weight: bold; } .block-hash { font-size: 0.8em; color: var(--text-light); word-break: break-all; }
        .transaction { font-size: 0.9em; margin-top: 5px; padding: 5px 10px; border-left: 2px solid var(--primary-color); background: #fff; border-radius: 4px;}
        #fl-status-box { background: #e3f2fd; border: 1px solid #90caf9; padding: 15px; border-radius: 6px; text-align: center; margin-bottom: 15px;}
        #fl-status-text { font-size: 1.2em; font-weight: bold; color: #1976d2; }
        #fl-participants { font-size: 0.9em; color: #42a5f5; margin-top: 10px; min-height: 1em; }
        #fl-details-table { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 10px; }
        #fl-details-table th, #fl-details-table td { border: 1px solid #ddd; padding: 6px; text-align: center; }
        #fl-details-table th { background-color: #e3f2fd; }
        #fl-details-table .global-row { font-weight: bold; background-color: #c5cae9; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>EV能源管理系统 (区块链 & 联邦学习)</h1>
        <div class="card">
            <div class="status-grid">
                <div class="status-item"><div class="label">模拟时间步</div><div class="value" id="sim-time">0</div></div>
                <div class="status-item"><div class="label">当前电价 (元/kWh)</div><div class="value" id="grid-price">0.00</div></div>
                <div class="status-item"><div class="label">电网负载</div><div class="value" id="grid-load">0%</div></div>
                <div class="status-item"><div class="label">区块数量</div><div class="value" id="block-count">0</div></div>
            </div>
        </div>

        <div class="main-grid">
            <div class="card">
                <h3>电动汽车 (EVs)</h3>
                <div id="ev-container" class="ev-container"></div>
            </div>
            <div class="card">
                <h3>联邦学习 (FL)</h3>
                <div id="fl-status-box">
                    <div id="fl-status-text">初始化...</div>
                    <div id="fl-participants"></div>
                </div>
                <div id="fl-details-container"></div>
                <canvas id="fl-chart" style="margin-top: 20px;"></canvas>
            </div>
            <div class="card">
                <h3>实时系统日志</h3>
                <div id="log-container" class="log-container"></div>
            </div>
            <div class="card" style="grid-column: span 2;">
                <h3>区块链浏览器</h3>
                <div id="blockchain-container" class="blockchain-container"></div>
            </div>
            <div class="card">
                <h3>待处理交易 (Mempool)</h3>
                <div id="pending-tx-container" class="blockchain-container"></div>
            </div>
        </div>
    </div>

    <script>
        let flChart;
        let lastBlockCount = 0;

        function initChart() {
            const ctx = document.getElementById('fl-chart').getContext('2d');
            flChart = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [{ label: '局部模型准确率', data: [], borderColor: 'var(--primary-color)', backgroundColor: 'rgba(63, 81, 181, 0.1)', fill: true, tension: 0.2 }] }, options: { scales: { y: { beginAtZero: true, max: 1.0 } }, animation: { duration: 500 } } });
        }

        function updateUI(data) {
            // 系统状态
            document.getElementById('sim-time').innerText = data.simulation_time;
            document.getElementById('grid-price').innerText = data.grid_price.toFixed(2);
            document.getElementById('grid-load').innerText = `${(data.grid_load * 100).toFixed(0)}%`;
            document.getElementById('block-count').innerText = data.blockchain.length;

            // EVs
            const evContainer = document.getElementById('ev-container');
            evContainer.innerHTML = '';
            data.evs.forEach(ev => {
                const evDiv = document.createElement('div');
                evDiv.className = `ev ${ev.is_training ? 'training' : ''}`;
                evDiv.innerHTML = `<div class="ev-id">EV #${ev.id}</div><div>SoC: ${(ev.soc * 100).toFixed(0)}%</div><div class="soc-bar"><div class="soc-fill" style="width: ${ev.soc * 100}%;"></div></div><div class="ev-status ${ev.status}">${ev.status}</div><div style="font-size:0.8em; color:#999;">Data: ${ev.local_data_points}</div>`;
                evContainer.appendChild(evDiv);
            });

            // 实时日志
            const logContainer = document.getElementById('log-container');
            logContainer.innerHTML = data.logs.map(log => `<div class="log-item"><span class="time">${log.time}</span> <span class="cat-${log.category}">${log.message}</span></div>`).join('');

            // 待处理交易
            const pendingTxContainer = document.getElementById('pending-tx-container');
            pendingTxContainer.innerHTML = '';
            if (data.pending_transactions.length === 0) {
                pendingTxContainer.innerHTML = '<div class="transaction" style="background:#f5f5f5; border-color:#eee;">等待新的交易...</div>';
            } else {
                data.pending_transactions.forEach(tx => {
                    const txDiv = document.createElement('div');
                    txDiv.className = 'transaction';
                    txDiv.style.background = '#fffde7';
                    txDiv.innerText = `EV #${tx.ev_id} ${tx.action} ${tx.amount_kwh} kWh`;
                    pendingTxContainer.appendChild(txDiv);
                });
            }

            // 区块链
            const blockchainContainer = document.getElementById('blockchain-container');
            const isNewBlock = data.blockchain.length > lastBlockCount;
            if (isNewBlock) {
                lastBlockCount = data.blockchain.length;
                blockchainContainer.innerHTML = ''; 
            }
            if (blockchainContainer.children.length !== data.blockchain.length) {
                blockchainContainer.innerHTML = '';
                [...data.blockchain].reverse().forEach((block, index) => {
                    const blockDiv = document.createElement('div');
                    blockDiv.className = 'block';
                    if (isNewBlock && index === 0) blockDiv.classList.add('new-block');
                    let transactionsHTML = block.transactions.map(tx => `<div class="transaction">EV #${tx.ev_id} -> ${tx.action}: ${tx.amount_kwh} kWh @ ${tx.price_per_kwh.toFixed(2)} 元</div>`).join('');
                    if (!transactionsHTML) transactionsHTML = '<div class="transaction"><i>创世区块 - 无交易</i></div>';
                    blockDiv.innerHTML = `<div class="block-header"><span>区块 #${block.index}</span><span>${new Date(block.timestamp * 1000).toLocaleTimeString()}</span></div><div><small>Hash: <span class="block-hash">${block.hash.substring(0, 20)}...</span></small></div>${transactionsHTML}`;
                    blockchainContainer.appendChild(blockDiv);
                });
            }
            
            // 联邦学习
            document.getElementById('fl-status-text').innerText = data.fl_status;
            const participantsDiv = document.getElementById('fl-participants');
            participantsDiv.innerText = data.fl_participants.length > 0 ? `参与者: EV #${data.fl_participants.join(', #')}` : '';
            
            const detailsContainer = document.getElementById('fl-details-container');
            if (data.fl_round_details && data.fl_round_details.aggregation) {
                const details = data.fl_round_details;
                let tableHTML = '<table id="fl-details-table"><tr><th>模型</th><th>权重 (SoC)</th><th>权重 (Price)</th><th>权重 (Time)</th></tr>';
                details.aggregation.local_weights.forEach((weights, i) => {
                    tableHTML += `<tr><td>EV #${details.participants[i]} (本地)</td><td>${weights[0].toFixed(2)}</td><td>${weights[1].toFixed(2)}</td><td>${weights[2].toFixed(2)}</td></tr>`;
                });
                const global_w = details.aggregation.global_weights;
                tableHTML += `<tr class="global-row"><td>全局 (聚合后)</td><td>${global_w[0].toFixed(2)}</td><td>${global_w[1].toFixed(2)}</td><td>${global_w[2].toFixed(2)}</td></tr>`;
                tableHTML += '</table>';
                detailsContainer.innerHTML = tableHTML;
            } else {
                detailsContainer.innerHTML = '';
            }

            // FL 图表
            flChart.data.labels = data.fl_history.map((_, i) => `R${i + 1}`);
            flChart.data.datasets[0].data = data.fl_history;
            flChart.update();
        }

        async function fetchData() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                updateUI(data);
            } catch (error) { console.error("无法获取数据:", error); }
        }

        window.onload = () => {
            initChart();
            fetchData();
            setInterval(fetchData, 1500);
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
    station.simulation_step()
    return jsonify(station.get_status())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)