from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx


class Edge:
    def __init__(self, u, v, cap):
        self.u = u
        self.v = v
        self.cap = cap
        self.flow = 0
        self.rev = None

    def remaining(self):
        return self.cap - self.flow


class FlowNetwork:
    def __init__(self):
        self.adj = defaultdict(list)
        self.edges = []

    def add_edge(self, u, v, cap):
        forward = Edge(u, v, cap)
        backward = Edge(v, u, 0)
        forward.rev = backward
        backward.rev = forward
        self.adj[u].append(forward)
        self.adj[v].append(backward)
        self.edges.append(forward)

    def reset_flow(self):
        for e in self.edges:
            e.flow = 0
            e.rev.flow = 0

    # 1. Edmonds-Karp (BFS)
    def edmonds_karp(self, s, t):
        self.reset_flow()
        max_flow = 0

        while True:
            parent_edge = {}
            queue = deque([s])

            while queue:
                curr = queue.popleft()
                if curr == t:
                    break
                for edge in self.adj[curr]:
                    if edge.remaining() > 0 and edge.v not in parent_edge and edge.v != s:
                        parent_edge[edge.v] = edge
                        queue.append(edge.v)

            # No augmenting path found
            if t not in parent_edge:
                break

            # Find bottleneck capacity
            push = float("inf")
            curr = t
            while curr != s:
                edge = parent_edge[curr]
                push = min(push, edge.remaining())
                curr = edge.u

            # Augment flow along the path
            curr = t
            while curr != s:
                edge = parent_edge[curr]
                edge.flow += push
                edge.rev.flow -= push
                curr = edge.u

            max_flow += push

        return max_flow

    # 2. Dinic's Algorithm (Level graph + blocking flow)
    def dinic(self, s, t):
        self.reset_flow()
        max_flow = 0

        def bfs_levels():
            levels = {s: 0}
            queue = deque([s])
            while queue:
                curr = queue.popleft()
                for edge in self.adj[curr]:
                    if edge.remaining() > 0 and edge.v not in levels:
                        levels[edge.v] = levels[curr] + 1
                        queue.append(edge.v)
            return levels

        def dfs_send(curr, flow_in, levels, ptr):
            if curr == t or flow_in == 0:
                return flow_in

            adj_list = self.adj[curr]
            while ptr[curr] < len(adj_list):
                edge = adj_list[ptr[curr]]
                if levels.get(edge.v) == levels[curr] + 1 and edge.remaining() > 0:
                    pushed = dfs_send(edge.v, min(flow_in, edge.remaining()), levels, ptr)
                    if pushed > 0:
                        edge.flow += pushed
                        edge.rev.flow -= pushed
                        return pushed
                ptr[curr] += 1
            return 0

        while True:
            levels = bfs_levels()
            if t not in levels:
                break
            ptr = defaultdict(int)
            while True:
                pushed = dfs_send(s, float("inf"), levels, ptr)
                if pushed == 0:
                    break
                max_flow += pushed

        return max_flow

    # Optional plot
    def plot(self, filename="flow_result.png"):
        g = nx.DiGraph()
        for e in self.edges:
            g.add_edge(e.u, e.v, flow=e.flow, cap=e.cap)

        pos = nx.spring_layout(g, seed=42)
        plt.figure(figsize=(8, 5))
        nx.draw_networkx_nodes(g, pos, node_color="#3b82f6", node_size=900)
        nx.draw_networkx_labels(g, pos, font_color="white", font_weight="bold")

        labels = {(u, v): f"{d['flow']}/{d['cap']}" for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edges(g, pos, arrowstyle="->", arrowsize=15, edge_color="#64748b")
        nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Graph image saved to {filename}")


if __name__ == "__main__":
    net = FlowNetwork()
    net.add_edge("s", "A", 10)
    net.add_edge("s", "C", 10)
    net.add_edge("A", "B", 4)
    net.add_edge("A", "C", 2)
    net.add_edge("A", "D", 8)
    net.add_edge("C", "D", 9)
    net.add_edge("B", "t", 10)
    net.add_edge("D", "t", 10)

    print("Edmonds-Karp max flow:", net.edmonds_karp("s", "t"))
    print("Dinic max flow:", net.dinic("s", "t"))
    net.plot()
