import csv

from network import ComplexNetwork


import csv

def load_network_from_csv(file_path):
    """
    从CSV文件加载有向网络
    
    参数:
        file_path (str): CSV文件路径
    
    返回:
        ComplexNetwork: 加载后的网络对象
    """
    # 创建有向网络
    network = ComplexNetwork(network_type="directed")
    
    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                # 跳过空行
                if not row:
                    continue
                    
                # 确保每行至少有3个值
                if len(row) < 3:
                    print(f"警告: 行 {row} 格式不正确，跳过")
                    continue
                    
                try:
                    source = int(row[0])
                    target = int(row[1])
                    edge_type = int(row[2])
                    
                    # 添加源节点和目标节点（如果不存在）
                    if source not in network.G:
                        network.add_node(source)
                    if target not in network.G:
                        network.add_node(target)
                    
                    # 根据边类型添加边
                    if edge_type == 1:
                        # 单向边: 源 -> 目标
                        network.add_edge(source, target)
                    elif edge_type == 2:
                        # 双向边: 源 <-> 目标
                        network.add_edge(source, target)
                        network.add_edge(target, source)
                    else:
                        print(f"警告: 未知边类型 {edge_type}，跳过")
                except ValueError as e:
                    print(f"警告: 无法将行 {row} 转换为整数: {e}")
    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 加载文件时发生意外错误: {e}")
        return None
    
    return network

if __name__ == "__main__":
    network = load_network_from_csv("profiles/network.csv")
    if network:
        # 显示网络基本信息
        print(f"节点数: {network.get_node_count()}")
        print(f"边数: {network.get_edge_count()}")
        print(f"聚类系数: {network.get_clustering_coefficient()}")
        print(f"平均最短路长度:{network.get_average_shortest_path_length()}")
        print(f"网络直径:{network.get_diameter()}")
        print(f"网络密度:{network.get_density()}")

        network.plot_degree_distribution(fit_power_law=True)
        
        
        network.save_network('profiles/net.gml')
