# src/ui.py
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def create_progress_bar():
    """创建一个标准化的 Rich 进度条管理器"""
    return Progress(
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True
    )
