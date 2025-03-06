import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from math import ceil
import warnings
warnings.filterwarnings('ignore')

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

class ChronosTrainer:
    def __init__(self, data_path, prediction_length=24, plot_history_length=84):
        """初始化训练器
        
        Args:
            data_path: 处理后的数据文件路径
            prediction_length: 预测长度（小时）
            plot_history_length: 绘图时显示的历史数据长度
        """
        self.prediction_length = prediction_length
        self.plot_history_length = plot_history_length
        self.data = None
        self.predictions = None
        self.model = None
        
        # 加载数据
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载并预处理数据"""
        try:
            # 读取数据
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 转换为TimeSeriesDataFrame格式
            self.data = TimeSeriesDataFrame(df)
            print(f"数据加载完成！")
            print(f"总记录数：{len(self.data)}")
            print(f"患者数量：{self.data.index.get_level_values('item_id').nunique()}")
            
        except Exception as e:
            print(f"数据加载失败：{str(e)}")
            self.data = None
    
    def train(self, **kwargs):
        """训练模型"""
        if self.data is None:
            print("错误：未成功加载数据！")
            return
        
        try:
            # 创建预测器
            self.model = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                eval_metric='MASE',
                **kwargs
            )
            
            # 训练模型
            self.model.fit(
                self.data,
                presets="medium_quality",
                time_limit=3600  # 1小时时间限制
            )
            
            # 生成预测
            self.predictions = self.model.predict(self.data)
            print("模型训练完成！")
            
        except Exception as e:
            print(f"训练失败：{str(e)}")
            self.model = None
            self.predictions = None
    
    def evaluate(self, save_plot=None):
        """评估模型性能并可视化结果
        
        Args:
            save_plot: 可选，图表保存路径
        """
        if self.predictions is None:
            print("错误：未进行预测！")
            return
        
        item_ids = self.data.index.get_level_values('item_id').unique()
        n_patients = len(item_ids)
        n_rows = ceil(n_patients / 2)
        n_cols = 2
        
        # 创建图表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        axes = axes.flatten()
        
        metrics = {
            'mae': [], 'mse': [], 'r2': [], 'pearson': []
        }
        
        for i, unique_id in enumerate(item_ids):
            # 获取该患者的数据
            patient_data = self.data[self.data.index.get_level_values('item_id') == unique_id]
            patient_predictions = self.predictions[self.predictions.index.get_level_values('item_id') == unique_id]
            test_patient_data = patient_data.iloc[-self.prediction_length:]
            
            # 计算指标
            metrics['mae'].append(mean_absolute_error(test_patient_data['target'], patient_predictions['mean']))
            metrics['mse'].append(mean_squared_error(test_patient_data['target'], patient_predictions['mean']))
            metrics['r2'].append(r2_score(test_patient_data['target'], patient_predictions['mean']))
            metrics['pearson'].append(pearsonr(test_patient_data['target'], patient_predictions['mean'])[0])
            
            # 绘制子图
            ax = axes[i]
            self._plot_patient_prediction(
                ax, patient_data, patient_predictions, unique_id,
                metrics['mae'][-1], metrics['r2'][-1], metrics['pearson'][-1]
            )
        
        # 隐藏多余的子图
        for j in range(n_patients, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot)
            print(f"图表已保存至：{save_plot}")
        else:
            plt.show()
        
        # 打印平均指标
        print("\n模型评估结果：")
        print(f"平均 MAE: {np.mean(metrics['mae']):.2f}")
        print(f"平均 MSE: {np.mean(metrics['mse']):.2f}")
        print(f"平均 R²: {np.mean(metrics['r2']):.2f}")
        print(f"平均 Pearson: {np.mean(metrics['pearson']):.2f}")
    
    def _plot_patient_prediction(self, ax, data, predictions, patient_id, mae, r2, pearson):
        """绘制单个患者的预测结果"""
        # 绘制历史数据
        ax.plot(
            data.index.get_level_values('timestamp')[-self.plot_history_length:],
            data['target'].values[-self.plot_history_length:],
            label=f'历史数据 (最近 {self.plot_history_length} 小时)',
            color='gray',
            alpha=0.6
        )
        
        # 绘制预测结果
        ax.plot(
            predictions.index.get_level_values('timestamp'),
            predictions['mean'],
            label='预测',
            color='red',
            linewidth=2
        )
        
        # 绘制置信区间
        if '0.1' in predictions.columns and '0.9' in predictions.columns:
            ax.fill_between(
                predictions.index.get_level_values('timestamp'),
                predictions['0.1'],
                predictions['0.9'],
                color='red',
                alpha=0.2,
                label='90% 置信区间'
            )
        
        # 设置标题和标签
        ax.set_title(
            f"患者: {patient_id}\n"
            f"MAE: {mae:.2f}, R²: {r2:.2f}, Pearson: {pearson:.2f}",
            fontsize=10
        )
        ax.set_xlabel('')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 添加图例
        if ax.get_subplotspec().is_first_col():
            ax.legend(loc='upper left') 