import json
import graphviz
from pathlib import Path
import textwrap

def create_graphviz_from_json(json_file):
    """
    Create a Graphviz visualization from the JSON structure
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='GDPR Argument Structure')
    dot.attr(rankdir='TB')  # Top to bottom layout (portrait)
    dot.attr('node', shape='ellipse', style='rounded,filled', fontsize='18')  # Larger font for nodes
    dot.attr('graph', splines='true', fontsize='16')  # Changed to 'true' for straight arrows
    dot.attr('edge', fontsize='12')  # Larger font for edges
    
    # Set page size and orientation for portrait
    dot.attr(size='8.5,5')  # Portrait page size (width, height)
    dot.attr(ratio='fill')   # Fill the page
    
    # Center the graph layout
    dot.attr(center='true')  # Center the graph on the page
    dot.attr(concentrate='true')  # Merge parallel edges
    
    def wrap_text(text, width=40):
        """Wrap text to specified width and join with newlines"""
        lines = textwrap.wrap(text, width=width)
        return '\n'.join(lines)
    
    def add_node_and_edges(parent_key, content, parent_id=None, level=0):
        """Recursively add nodes and edges to the graph"""
        
        # Color scheme based on level
        colors = {
            0: '#F5F5F5',  # Light gray - Main Claim (root)
            1: '#F5F5F5',  # Light gray - Sub Claims
            2: '#F5F5F5',  # Light gray - Argument Claims
            3: '#F5F5F5',  # Light gray - Argument Sub Claims
            4: '#DAE8FC'   # Light blue - Evidence
        }
        
        if isinstance(content, dict):
            if 'description' in content:
                # Create node ID
                node_id = f"{parent_key}_{id(content)}"
                
                # Wrap description text instead of truncating
                desc = content['description']
                # Limit total length but wrap instead of truncate
                if len(desc) > 200:  # Limit total characters
                    desc = desc[:197] + "..."
                
                # Wrap the text to fit in the node
                wrapped_desc = wrap_text(desc, width=35)  # Adjust width as needed
                
                # Add node with appropriate styling and larger font
                color = colors.get(level, '#F0F0F0')
                
                # Special styling for root node (level 0)
                if level == 0:
                    dot.node(node_id, f"{parent_key}\n{wrapped_desc}", 
                            fillcolor=color, 
                            fontsize='22',  # Larger font for root
                            width='5',      # Wider root node
                            height='3',     # Taller root node
                            style='rounded,filled,bold')  # Bold border for root
                else:
                    dot.node(node_id, f"{parent_key}\n{wrapped_desc}", 
                            fillcolor=color, 
                            fontsize='20',  # Larger font size
                            width='4',      # Wider nodes to accommodate text
                            height='2.5')   # Taller nodes
                
                # Add edge from parent if exists
                if parent_id:
                    dot.edge(parent_id, node_id, fontsize='20')  # Larger edge font
                
                # Process children
                for key, value in content.items():
                    if key != 'description':
                        add_node_and_edges(key, value, node_id, level + 1)
                        
            else:
                # Handle containers without descriptions
                for key, value in content.items():
                    add_node_and_edges(key, value, parent_id, level)
        
        elif isinstance(content, list):
            # Handle lists (like SubClaims, ArgumentClaims, etc.)
            for i, item in enumerate(content):
                add_node_and_edges(f"{parent_key}_{i+1}", item, parent_id, level)
    
    # Start processing from the root
    add_node_and_edges("MainClaim", data["MainClaim"])
    
    return dot

def save_graphviz_outputs(dot, base_filename):
    """Save Graphviz output in multiple formats"""
    # Save as DOT file
    with open(f"{base_filename}.dot", 'w') as f:
        f.write(dot.source)
    
    # Render as PDF
    dot.render(base_filename, format='pdf', cleanup=True)
    
    # Render as PNG with higher DPI for better quality
    dot.render(base_filename, format='png', cleanup=True)
    
    # Render as SVG
    dot.render(base_filename, format='svg', cleanup=True)
    
    print(f"Generated files:")
    print(f"  - {base_filename}.dot")
    print(f"  - {base_filename}.pdf")
    print(f"  - {base_filename}.png")
    print(f"  - {base_filename}.svg")

# Usage
if __name__ == "__main__":
    import sys
    json_file = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-14B-Instruct_R23_1.json"

    # Create Graphviz visualization
    dot = create_graphviz_from_json(json_file)
    
    # Save outputs
    base_name = Path(json_file).stem + "_graphviz"
    save_graphviz_outputs(dot, base_name)