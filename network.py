import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class ComplexNetwork:
    """表示和分析复杂网络的类"""
    
    def __init__(self, network_type="undirected"):
        """
        初始化复杂网络对象
        
        参数:
            network_type (str): 网络类型，"undirected"（无向网络）或"directed"（有向网络）
        """
        if network_type == "undirected":
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        self.network_type = network_type
    
    def add_node(self, node_id, **attr):
        """添加节点到网络"""
        self.G.add_node(node_id, **attr)
    
    def add_nodes(self, node_list):
        """批量添加节点到网络"""
        for node in node_list:
            self.add_node(node)
    
    def add_edge(self, source, target, **attr):
        """添加边到网络"""
        self.G.add_edge(source, target, **attr)
    
    def add_edges(self, edge_list):
        """批量添加边到网络"""
        for edge in edge_list:
            if len(edge) == 2:
                source, target = edge
                self.add_edge(source, target)
            elif len(edge) == 3:
                source, target, attr = edge
                self.add_edge(source, target, **attr)
    
    def remove_node(self, node_id):
        """从网络中移除节点"""
        if node_id in self.G:
            self.G.remove_node(node_id)
    
    def remove_edge(self, source, target):
        """从网络中移除边"""
        if self.G.has_edge(source, target):
            self.G.remove_edge(source, target)
    
    def get_node_count(self):
        """获取网络中的节点数量"""
        return self.G.number_of_nodes()
    
    def get_edge_count(self):
        """获取网络中的边数量"""
        return self.G.number_of_edges()
    
    def get_degree(self, node_id=None):
        """
        获取节点的度（无向网络）或入度和出度（有向网络）
        
        参数:
            node_id (optional): 节点ID，若为None则返回所有节点的度
        
        返回:
            节点的度（无向网络）或入度和出度（有向网络）
        """
        if node_id is not None:
            if self.network_type == "undirected":
                return self.G.degree(node_id)
            else:
                return self.G.in_degree(node_id), self.G.out_degree(node_id)
        else:
            if self.network_type == "undirected":
                return dict(self.G.degree())
            else:
                return dict(self.G.in_degree()), dict(self.G.out_degree())
    
    def get_density(self):
        """计算网络的密度"""
        return nx.density(self.G)
    
    def get_clustering_coefficient(self, node_id=None):
        """
        计算聚类系数
        
        参数:
            node_id (optional): 节点ID，若为None则计算整个网络的平均聚类系数
        
        返回:
            节点的聚类系数或整个网络的平均聚类系数
        """
        if node_id is not None:
            return nx.clustering(self.G, node_id)
        else:
            return nx.average_clustering(self.G)
    
    def get_shortest_path(self, source, target):
        """计算两个节点之间的最短路径"""
        if nx.has_path(self.G, source, target):
            return nx.shortest_path(self.G, source, target)
        else:
            return None
    
    def get_average_shortest_path_length(self):
        """计算网络的平均最短路径长度"""
        if nx.is_connected(self.G) if self.network_type == "undirected" else nx.is_strongly_connected(self.G):
            return nx.average_shortest_path_length(self.G)
        else:
            return None
    
    def get_diameter(self):
        """计算网络的直径"""
        if nx.is_connected(self.G) if self.network_type == "undirected" else nx.is_strongly_connected(self.G):
            return nx.diameter(self.G)
        else:
            return None
    
    def get_degree_distribution(self, normalized=True):
        """
        计算度分布
        
        参数:
            normalized (bool): 是否归一化
        
        返回:
            度分布字典，键为度，值为对应频率
        """
        degrees = [d for _, d in self.G.degree()]
        degree_counts = Counter(degrees)
        
        if normalized:
            total = sum(degree_counts.values())
            return {k: v / total for k, v in degree_counts.items()}
        else:
            return dict(degree_counts)
    
    def plot_degree_distribution(self, log_scale=True, title="Degree Distribution", 
                                 fit_power_law=False, bins=None, ax=None):
        """
        绘制度分布
        
        参数:
            log_scale (bool): 是否使用对数坐标
            title (str): 图表标题
            fit_power_law (bool): 是否拟合幂律分布
            bins (int or str): 直方图分箱方式，传递给numpy.histogram
            ax (matplotlib.axes.Axes): 可选的matplotlib轴对象
        """
        # 获取度分布
        degrees = [d for _, d in self.G.degree()]
        
        # 创建图表
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制度分布
        if bins is not None:
            # 使用直方图方式绘制
            hist, bin_edges = np.histogram(degrees, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            # 过滤零值以避免对数问题
            mask = hist > 0
            ax.plot(bin_centers[mask], hist[mask], 'o', markersize=8)
        else:
            # 使用散点图方式绘制
            degree_counts = Counter(degrees)
            x = np.array(sorted(degree_counts.keys()))
            y = np.array([degree_counts[k] for k in x])
            y = y / np.sum(y)  # 归一化
            
            ax.plot(x, y, 'o', markersize=8)
        
        # 添加幂律拟合线
        if fit_power_law and len(degrees) > 10:
            from scipy.optimize import curve_fit
            
            def power_law(x, a, b):
                return a * np.power(x, b)
            
            # 过滤零值并取对数
            x_data = np.array([d for d in degrees if d > 0])
            y_data, _ = np.histogram(x_data, bins='auto', density=True)
            x_bin_centers = 0.5 * (_[1:] + _[:-1])
            x_bin_centers = x_bin_centers[y_data > 0]
            y_data = y_data[y_data > 0]
            
            # 拟合幂律分布
            try:
                popt, _ = curve_fit(power_law, x_bin_centers, y_data, p0=[1, -2])
                x_fit = np.logspace(np.log10(min(x_bin_centers)), np.log10(max(x_bin_centers)), 100)
                y_fit = power_law(x_fit, *popt)
                ax.plot(x_fit, y_fit, 'r--', 
                        label=f'Power Law Fit: $y = {popt[0]:.4f}x^{{{popt[1]:.4f}}}$')
            except:
                print("警告: 幂律拟合失败")
        
        # 设置坐标轴和标题
        ax.set_xlabel('Degree (k)')
        ax.set_ylabel('Probability P(k)')
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 设置对数坐标
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # 添加图例
        if fit_power_law:
            ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 如果没有提供轴对象，则显示图表
        if ax is None:
            plt.show()
    
    def visualize(self, title="Complex Network Visualization", layout="spring", node_size=300, 
                  node_color='skyblue', edge_color='gray', with_labels=True):
        """
        可视化网络
        
        参数:
            title (str): 图表标题
            layout (str): 布局算法，可选'spring', 'circular', 'random', 'shell', 'spectral'
            node_size (int): 节点大小
            node_color (str): 节点颜色
            edge_color (str): 边颜色
            with_labels (bool): 是否显示节点标签
        """
        plt.figure(figsize=(10, 8))
        
        # 选择布局算法
        if layout == "spring":
            pos = nx.spring_layout(self.G)
        elif layout == "circular":
            pos = nx.circular_layout(self.G)
        elif layout == "random":
            pos = nx.random_layout(self.G)
        elif layout == "shell":
            pos = nx.shell_layout(self.G)
        elif layout == "spectral":
            pos = nx.spectral_layout(self.G)
        else:
            pos = nx.spring_layout(self.G)
        
        # 绘制网络
        nx.draw(
            self.G, 
            pos, 
            node_size=node_size, 
            node_color=node_color, 
            edge_color=edge_color,
            with_labels=with_labels,
            font_size=10,
            alpha=0.8
        )
        
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def save_network(self, filename):
        """将网络保存到文件"""
        if filename.endswith('.gml'):
            nx.write_gml(self.G, filename)
        elif filename.endswith('.graphml'):
            nx.write_graphml(self.G, filename)
        elif filename.endswith('.gexf'):
            nx.write_gexf(self.G, filename)
        else:
            nx.write_edgelist(self.G, filename)
    
    @classmethod
    def load_network(cls, filename, network_type="undirected"):
        """从文件加载网络"""
        obj = cls(network_type)
        
        if filename.endswith('.gml'):
            obj.G = nx.read_gml(filename)
        elif filename.endswith('.graphml'):
            obj.G = nx.read_graphml(filename)
        elif filename.endswith('.gexf'):
            obj.G = nx.read_gexf(filename)
        else:
            if network_type == "undirected":
                obj.G = nx.read_edgelist(filename)
            else:
                obj.G = nx.read_edgelist(filename, create_using=nx.DiGraph())
        
        obj.network_type = network_type
        return obj

# 示例使用
if __name__ == "__main__":
    # 创建一个无向网络
    network = ComplexNetwork()
    
    # 添加节点
    network.add_nodes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 添加边
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), 
             (5, 7), (6, 7), (6, 8), (7, 9), (8, 9), (8, 10), (9, 10)]
    network.add_edges(edges)
    
    # 显示网络基本信息
    print(f"节点数: {network.get_node_count()}")
    print(f"边数: {network.get_edge_count()}")
    print(f"密度: {network.get_density():.4f}")
    print(f"平均聚类系数: {network.get_clustering_coefficient():.4f}")
    print(f"平均最短路径长度: {network.get_average_shortest_path_length():.4f}")
    print(f"直径: {network.get_diameter()}")
    
    # 可视化网络
    network.visualize(title="示例复杂网络")
    
    # 绘制度分布
    network.plot_degree_distribution(title="示例网络度分布")    