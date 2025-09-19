import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Example JSON (shortened — you can load from file instead)
data = {
    "Claim": {
        "description": "The system enforces least privilege across all components.",
        "block_type": "Decomposition",
        "final_status": "khaki",
        "SubClaims": [
            {
                "SubClaim1": {
                    "description": "Access policies are in place to restrict user permissions.",
                    "block_type": "Concretion",
                    "final_status": "green",
                    "Arguments": [
                        {
                            "Argument1": {
                                "description": "Policies and system design enforce role-based restrictions.",
                                "block_type": "Evidence-incorporation",
                                "final_status": "green",
                                "Evidences": [
                                    {
                                        "Evidence1": {
                                            "description": "System design document specifying user roles.",
                                            "type": "document",
                                            "edge_scores": [
                                                {
                                                    "parent": "Argument1",
                                                    "comprehensiveness_score": 0.86,
                                                    "sufficiency_score": 0.81
                                                }
                                            ]
                                        }
                                    },
                                    {
                                        "Evidence2": {
                                            "description": "Policy file defining access control rules.",
                                            "type": "policy",
                                            "edge_scores": [
                                                {
                                                    "parent": "Argument1",
                                                    "comprehensiveness_score": 0.90,
                                                    "sufficiency_score": 0.84
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "SubClaim2": {
                    "description": "Audit logs demonstrate least privilege in operation.",
                    "block_type": "Concretion",
                    "final_status": "red",
                    "Arguments": [
                        {
                            "Argument2": {
                                "description": "Audit evidence confirms enforcement of least privilege.",
                                "block_type": "Evidence-incorporation",
                                "final_status": "red",
                                "Evidences": [
                                    {
                                        "Evidence3": {
                                            "description": "Audit log of user access requests.",
                                            "type": "log",
                                            "edge_scores": [
                                                {
                                                    "parent": "Argument2",
                                                    "comprehensiveness_score": 0.62,
                                                    "sufficiency_score": 0.55
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}

# ------------------------------
# Recursive function to build tree graph with path tracing
# ------------------------------
def add_nodes_edges(graph, parent_name, node_dict, level=0, path_scores=None):
    """Recursively add nodes and edges to create a tree structure with path-based scoring."""
    if path_scores is None:
        path_scores = {'comprehensiveness': 1.0, 'sufficiency': 1.0}
    
    for key, value in node_dict.items():
        if isinstance(value, dict):
            node_name = key
            status = value.get("final_status", "gray")
            block_type = value.get("block_type", "")
            description = value.get("description", "")
            
            # Add node with attributes including cumulative path scores
            graph.add_node(
                node_name, 
                color=status, 
                description=description,
                block_type=block_type,
                level=level,
                path_comp_score=path_scores['comprehensiveness'],
                path_suff_score=path_scores['sufficiency']
            )
            
            if parent_name:
                # Add edge with cumulative path scores
                graph.add_edge(parent_name, node_name)
                graph[parent_name][node_name]['path_comp_score'] = path_scores['comprehensiveness']
                graph[parent_name][node_name]['path_suff_score'] = path_scores['sufficiency']
                # Shorter labels for compact display
                graph[parent_name][node_name]['label'] = f"C:{path_scores['comprehensiveness']:.2f}|S:{path_scores['sufficiency']:.2f}"
            
            # Process SubClaims (inherit path scores)
            if "SubClaims" in value:
                for sub in value["SubClaims"]:
                    add_nodes_edges(graph, node_name, sub, level + 1, path_scores.copy())
            
            # Process Arguments (inherit path scores)
            if "Arguments" in value:
                for arg in value["Arguments"]:
                    add_nodes_edges(graph, node_name, arg, level + 1, path_scores.copy())
            
            # Process Evidences (multiply path scores with evidence scores)
            if "Evidences" in value:
                for ev in value["Evidences"]:
                    for ev_key, ev_val in ev.items():
                        # Calculate new path scores for evidence
                        evidence_path_scores = path_scores.copy()
                        
                        if "edge_scores" in ev_val and ev_val["edge_scores"]:
                            scores = ev_val["edge_scores"][0]
                            comp_score = scores.get('comprehensiveness_score', 1.0)
                            suff_score = scores.get('sufficiency_score', 1.0)
                            
                            # Multiply path scores (representing path strength)
                            evidence_path_scores['comprehensiveness'] *= comp_score
                            evidence_path_scores['sufficiency'] *= suff_score
                        
                        # Add evidence node with final path scores
                        graph.add_node(
                            ev_key, 
                            color="lightblue", 
                            description=ev_val.get("description", ""),
                            block_type="Evidence",
                            level=level + 1,
                            path_comp_score=evidence_path_scores['comprehensiveness'],
                            path_suff_score=evidence_path_scores['sufficiency']
                        )
                        
                        # Add edge with final path scores
                        graph.add_edge(node_name, ev_key)
                        graph[node_name][ev_key]['path_comp_score'] = evidence_path_scores['comprehensiveness']
                        graph[node_name][ev_key]['path_suff_score'] = evidence_path_scores['sufficiency']
                        
                        # Create compact path tracing label
                        if "edge_scores" in ev_val and ev_val["edge_scores"]:
                            scores = ev_val["edge_scores"][0]
                            local_comp = scores.get('comprehensiveness_score', 1.0)
                            local_suff = scores.get('sufficiency_score', 1.0)
                            # Very compact format
                            graph[node_name][ev_key]['label'] = (
                                f"Compr: {local_comp:.2f},Suff: {local_suff:.2f}"
                                # f"P:Compr{evidence_path_scores['comprehensiveness']:.2f},Suff{evidence_path_scores['sufficiency']:.2f}"
                            )
                        else:
                            graph[node_name][ev_key]['label'] = (
                                f"Compr: {evidence_path_scores['comprehensiveness']:.2f},Suff: {evidence_path_scores['sufficiency']:.2f}"
                            )

def get_path_to_root(graph, node):
    """Get the path from a node back to the root."""
    path = [node]
    current = node
    
    # Traverse back to root
    while True:
        predecessors = list(graph.predecessors(current))
        if not predecessors:
            break
        current = predecessors[0]  # Assuming tree structure (single parent)
        path.append(current)
    
    return list(reversed(path))  # Return path from root to node

def calculate_path_metrics(graph):
    """Calculate and display path metrics for all evidence nodes."""
    evidence_nodes = [n for n in graph.nodes() if graph.nodes[n].get('block_type') == 'Evidence']
    
    print("\n=== Path Tracing Analysis ===")
    for evidence in evidence_nodes:
        path = get_path_to_root(graph, evidence)
        path_comp = graph.nodes[evidence].get('path_comp_score', 1.0)
        path_suff = graph.nodes[evidence].get('path_suff_score', 1.0)
        
        print(f"Evidence {evidence}: Path {' → '.join(path)}")
        # print(f"  Final: C={path_comp:.3f}, S={path_suff:.3f}, Strength={path_comp * path_suff:.3f}")
        print(f"  Final: C={path_comp:.3f}, S={path_suff:.3f}")

def create_compact_tree_layout(G, root='Claim', width=6, vert_gap=1.0):
    """
    Create a compact hierarchical tree layout for document attachment.
    """
    # Get hierarchy levels using BFS
    levels = {}
    level_widths = {}
    queue = [(root, 0)]
    visited = {root}
    
    # Build level structure
    while queue:
        node, level = queue.pop(0)
        levels[node] = level
        
        if level not in level_widths:
            level_widths[level] = []
        level_widths[level].append(node)
        
        # Add children to queue
        children = list(G.successors(node))
        for child in children:
            if child not in visited:
                visited.add(child)
                queue.append((child, level + 1))
    
    # Calculate positions with tighter spacing
    pos = {}
    max_level = max(levels.values()) if levels else 0
    
    # Position nodes level by level
    for level in range(max_level + 1):
        nodes_at_level = level_widths.get(level, [])
        num_nodes = len(nodes_at_level)
        
        if num_nodes == 0:
            continue
            
        # Calculate Y position (top-down) with tighter spacing
        y = -level * vert_gap
        
        # Calculate X positions with adaptive width
        if num_nodes == 1:
            x_positions = [0]
        else:
            # Adaptive width based on number of nodes
            actual_width = min(width, num_nodes * 1.5)
            spacing = actual_width / max(1, num_nodes - 1) if num_nodes > 1 else 0
            start_x = -actual_width / 2
            x_positions = [start_x + i * spacing for i in range(num_nodes)]
        
        # Assign positions
        for i, node in enumerate(nodes_at_level):
            pos[node] = (x_positions[i], y)
    
    return pos

def get_node_shapes_and_colors(G):
    """Define node shapes and colors based on block type and status."""
    color_map = {
        "green": "lightgreen",
        "khaki": "khaki", 
        "red": "lightcoral",
        "gray": "lightgray",
        "lightblue": "lightblue"
    }
    
    shape_map = {
        "Decomposition": "s",  # square
        "Concretion": "o",     # circle
        "Evidence-incorporation": "^",  # triangle
        "Evidence": "D"        # diamond
    }
    
    node_colors = []
    node_shapes = []
    
    for node in G.nodes():
        color = G.nodes[node].get('color', 'gray')
        node_colors.append(color_map.get(color, 'lightgray'))
        
        block_type = G.nodes[node].get('block_type', '')
        node_shapes.append(shape_map.get(block_type, 'o'))
    
    return node_colors, node_shapes

def draw_compact_tree(G, pos, node_colors, node_shapes):
    """Draw a compact tree suitable for document attachment."""
    # Smaller figure size for document attachment
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Draw nodes with smaller size
    unique_shapes = set(node_shapes)
    for shape in unique_shapes:
        node_list = [node for i, node in enumerate(G.nodes()) if node_shapes[i] == shape]
        colors = [node_colors[i] for i, node in enumerate(G.nodes()) if node_shapes[i] == shape]
        
        if node_list:
            nx.draw_networkx_nodes(
                G, pos, nodelist=node_list, 
                node_color=colors, node_shape=shape,
                node_size=1200, alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax
            )
    
    # Draw edges with simplified styling
    for edge in G.edges():
        path_strength = (G[edge[0]][edge[1]].get('path_comp_score', 1.0) * 
                        G[edge[0]][edge[1]].get('path_suff_score', 1.0))
        edge_width = max(1, path_strength * 3)  # Thinner edges
        
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='gray', 
                              arrows=True, arrowsize=15, arrowstyle='->', 
                              width=edge_width, alpha=0.7, ax=ax)
    
    # Draw compact node labels
    node_labels = {}
    for node in G.nodes():
        # Abbreviated node names for compact display
        if 'SubClaim' in node:
            node_labels[node] = node.replace('SubClaim', 'SC')
        elif 'Argument' in node:
            node_labels[node] = node.replace('Argument', 'A')
        elif 'Evidence' in node:
            node_labels[node] = node.replace('Evidence', 'E')
        else:
            node_labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, 
                           font_weight="bold", font_color='black', ax=ax)
    
    # Draw compact edge labels
    edge_labels = {}
    for edge in G.edges():
        if 'label' in G[edge[0]][edge[1]]:
            label = G[edge[0]][edge[1]]['label']
            edge_labels[edge] = label
    
    if edge_labels:
        # Smaller, more compact edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                   font_size=6, bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor="lightyellow", alpha=0.8), ax=ax)

def create_compact_legend():
    """Create a compact legend for document attachment."""
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                   markersize=8, label='Decomposition', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=8, label='Concretion', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                   markersize=8, label='Evidence-inc', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='lightblue', 
                   markersize=8, label='Evidence', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markersize=8, label='Green', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='khaki', 
                   markersize=8, label='Khaki', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=8, label='Red', markeredgecolor='black')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
              frameon=True, fontsize=8, ncol=2)

# ------------------------------
# Main visualization function
# ------------------------------
def visualize_compact_assurance_case(data, title="Assurance Case Tree"):
    """Create a compact visualization suitable for document attachment."""
    
    # Build the graph with path tracing
    G = nx.DiGraph()
    add_nodes_edges(G, None, data)
    
    # Calculate and display path metrics (shorter output)
    calculate_path_metrics(G)
    
    # Create compact tree layout
    print("\nCreating compact tree layout...")
    pos = create_compact_tree_layout(G, root='Claim', width=5, vert_gap=0.8)
    
    # Get node properties
    node_colors, node_shapes = get_node_shapes_and_colors(G)
    
    # Draw the compact tree
    draw_compact_tree(G, pos, node_colors, node_shapes)
    
    # Add compact legend
    create_compact_legend()
    
    # Compact formatting
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    
    # Brief tree info
    print(f"Tree: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"depth {max([G.nodes[n].get('level', 0) for n in G.nodes()]) + 1}")

# ------------------------------
# Execute visualization
# ------------------------------
if __name__ == "__main__":
    # Visualize the compact assurance case
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "../ex3/out"
    os.makedirs(output_dir, exist_ok=True)
    visualize_compact_assurance_case(data)

    # Save with higher DPI for document quality
    plt.savefig(os.path.join(output_dir, "assurance_case_compact.png"), 
                format="png", dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "assurance_case_compact.pdf"), 
                format="pdf", bbox_inches='tight')  # PDF for better document integration
    
    print(f"\nCompact visualization saved to:")
    print(f"- PNG: {os.path.join(output_dir, 'assurance_case_compact.png')}")
    print(f"- PDF: {os.path.join(output_dir, 'assurance_case_compact.pdf')}")
    plt.show()