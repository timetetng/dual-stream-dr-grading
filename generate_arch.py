import os
import torch
import traceback
from torchview import draw_graph
from rich.console import Console

# 直接导入你项目中真实的完全体双流网络
from src.models.fusion import DualStreamNet

console = Console()

def generate_real_architecture_diagram():
    console.print("\n[bold cyan]>>> 正在解析工程代码，生成真实双流网络架构图...[/bold cyan]")
    
    save_dir = "outputs/reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    filename = "aptos_dual_stream_arch"
    save_path = os.path.join(save_dir, filename)

    model = DualStreamNet(
        num_classes=5, 
        embed_dim=512, 
        use_freq=True, 
        fusion_type='gated', 
        freq_type='math_prior'
    )
    model.eval()

    try:
        # 1. save_graph=False 避免 torchview 内部吞掉真实的报错信息
        # 2. depth=2 是画 ResNet 类网络最完美的深度，既有结构又不至于让渲染引擎崩溃
        model_graph = draw_graph(
            model, 
            input_size=(1, 3, 384, 384),
            graph_name=filename,
            depth=2,  
            expand_nested=True,
            show_shapes=True,
            show_dtypes=False,
            save_graph=False 
        )
        
        # 我们手动调用 graphviz 的 render 方法，以便暴露真实的底层报错
        console.print("[dim]正在调用 Graphviz 引擎渲染 PDF...[/dim]")
        model_graph.visual_graph.render(save_path, format="pdf", cleanup=True)
        
        console.print("[dim]正在调用 Graphviz 引擎渲染 PNG...[/dim]")
        model_graph.visual_graph.render(save_path, format="png", cleanup=True)
        
        console.print(f"[bold green]架构图生成成功！[/bold green]")
        console.print(f"  - 论文高清矢量图: {save_path}.pdf")
        console.print(f"  - 快速预览图:     {save_path}.png")
        
    except Exception as e:
        console.print("[bold red]生成失败！真实的底层报错信息如下：[/bold red]")
        traceback.print_exc()

if __name__ == "__main__":
    generate_real_architecture_diagram()
