import typer
import httpx
import os
from pathlib import Path
from typing import Optional, List, Tuple
from typing_extensions import Annotated

app = typer.Typer(help="CircuitSight 全流程测试客户端")

def get_images_from_dir(directory: Path) -> List[Tuple[str, str, str]]:
    files = []
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    if not directory.exists(): return []
    for img_path in directory.glob("*.*"):
        if img_path.suffix.lower() in valid_exts:
            files.append((img_path.name, str(img_path)))
    return files

def generate_html_report(data: dict, output_file: Path, query_dir: Path):
    total = data.get("total_processed", 0)

    css_style = """
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 20px; background: #eaeff2; color: #333; }
        .container { max-width: 1300px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
        h1 { border-bottom: 3px solid #3f51b5; padding-bottom: 15px; color: #283593; margin-bottom: 30px; }
        
        .stats { display: flex; gap: 30px; margin-bottom: 40px; background: #e8eaf6; padding: 20px; border-radius: 12px; color: #3949ab; font-weight: 500; }
        
        .global-viz { text-align: center; margin: 40px 0; padding: 30px; background: #fafafa; border: 1px solid #eee; border-radius: 12px; }
        .global-viz img { max-width: 900px; width: 100%; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        
        /* Grid Layout */
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 30px; }
        
        .card { border: 0; border-radius: 12px; overflow: hidden; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.08); transition: transform 0.2s; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }
        
        .card-header { background: #3f51b5; color: white; padding: 15px 20px; font-weight: bold; font-size: 1.1em; display: flex; align-items: center; gap: 10px; }
        .card-body { padding: 20px; }
        
        /* Match Item Layout */
        .match-item { 
            display: flex;
            align-items: flex-start;
            padding: 15px 0; 
            border-bottom: 1px solid #eee; 
            gap: 20px;
        }
        .match-item:last-child { border-bottom: none; }
        
        .rank-badge { 
            background: #e0e0e0; color: #555; 
            width: 28px; height: 28px; 
            border-radius: 50%; 
            display: flex; align-items: center; justify-content: center; 
            font-weight: bold; font-size: 0.9em; flex-shrink: 0;
        }
        .rank-1 { background: #ffd700; color: #795548; }
        
        .info-col { flex: 1; display: flex; flex-direction: column; gap: 5px; }
        .label-name { font-size: 1.2em; font-weight: bold; color: #333; }
        
        .metrics-row { display: flex; gap: 15px; margin-top: 5px; font-size: 0.9em; }
        .metric { background: #f5f5f5; padding: 4px 8px; border-radius: 6px; }
        .metric-label { color: #777; margin-right: 4px; }
        .metric-val { font-family: monospace; font-weight: bold; }
        
        .struct-score { color: #1565c0; font-size: 1.1em; }
        .struct-score.high { color: #2e7d32; }
        
        .heatmap-col { flex-shrink: 0; width: 240px; display: flex; flex-direction: column; align-items: center; }
        .heatmap-img { width: 100%; height: auto; border-radius: 6px; border: 1px solid #ddd; }
        .heatmap-caption { font-size: 0.75em; color: #999; margin-top: 4px; }
        
        .tag { padding: 4px 10px; border-radius: 20px; font-size: 0.8em; font-weight: bold; text-transform: uppercase; display: inline-block; }
        .tag-trusted { background: #e8f5e9; color: #2e7d32; }
        .tag-untrusted { background: #ffebee; color: #c62828; }
        
        .viz-chart-img { width: 100%; margin-top: 20px; border-radius: 8px; border: 1px solid #eee; }
    </style>
    """

    html_parts = []
    html_parts.append(f"""
    <!DOCTYPE html>
    <html><head><title>CircuitSight Report</title><meta charset="utf-8">{css_style}</head>
    <body>
    <div class="container">
        <h1>⚡ CircuitSight 深度识别报告</h1>
        <div class="stats">
            <div>📂 测试目录: {query_dir}</div>
            <div>🖼️ 图片数量: {total}</div>
        </div>
    """)

    if data.get("global_visualization"):
        html_parts.append(f"""
        <div class="global-viz">
            <h3>📐 全局向量空间分布 (ConvNeXt Embedding)</h3>
            <p><small>蓝色: 规则库 | 红色: 待测图</small></p>
            <img src="data:image/png;base64,{data['global_visualization']}" />
        </div>
        """)

    html_parts.append("<h2>🔍 识别详情 (结构优先匹配)</h2><div class='grid'>")
    
    for item in data["results"]:
        matches_html = ""
        best_trusted = False
        
        for i, m in enumerate(item["matches"]):
            rank_cls = "rank-1" if i == 0 else ""
            
            # 数据解包
            label = m['label']
            vec_dist = m['vector_dist']
            struct_score = m['structure_score']
            is_trusted = m['trusted']
            heatmap = m['heatmap']
            
            if i == 0 and is_trusted: best_trusted = True
            
            status_tag = f"<span class='tag tag-trusted'>匹配成功</span>" if is_trusted else f"<span class='tag tag-untrusted'>存疑</span>"
            
            heatmap_html = ""
            if heatmap:
                heatmap_html = f"""
                <div class="heatmap-col">
                    <img src="data:image/png;base64,{heatmap}" class="heatmap-img" />
                    <div class="heatmap-caption">结构对齐热力图</div>
                </div>
                """
            
            matches_html += f"""
            <div class="match-item">
                <div class="rank-badge {rank_cls}">{i+1}</div>
                
                <div class="info-col">
                    <div class="label-name">{label}</div>
                    <div>{status_tag}</div>
                    
                    <div class="metrics-row">
                        <div class="metric" title="ConvNeXt 提取的特征距离 (越小越像)">
                            <span class="metric-label">向量距离:</span>
                            <span class="metric-val">{vec_dist:.3f}</span>
                        </div>
                        <div class="metric" title="Grid HOG 结构匹配分数 (越高越像)">
                            <span class="metric-label">结构相似度:</span>
                            <span class="metric-val struct-score {'high' if struct_score>60 else ''}">{struct_score:.1f}%</span>
                        </div>
                    </div>
                </div>
                
                {heatmap_html}
            </div>
            """

        header_icon = "✅" if best_trusted else "⚠️"
        viz_img = ""
        if item["visualization"]:
            viz_img = f'<img class="viz-chart-img" src="data:image/png;base64,{item["visualization"]}" />'

        html_parts.append(f"""
        <div class="card">
            <div class="card-header">
                <span>{header_icon} {item['filename']}</span>
            </div>
            <div class="card-body">
                {matches_html}
                {viz_img}
            </div>
        </div>
        """)

    html_parts.append("</div></div></body></html>")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"\n✅ 报告已生成: {output_file.absolute()}")

# ... main 函数保持不变 (和上一个版本一致) ...
@app.command()
def main(
    rules_dir: Annotated[Optional[Path], typer.Option("--rules-dir", "-r")] = None,
    clear: Annotated[bool, typer.Option("--clear", "-c")] = False,
    query_dir: Annotated[Path, typer.Option("--query-dir", "-q")] = Path("./data/test_images"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("report.html"),
    host: str = "localhost",
    port: int = 8000,
    top_n: int = 3,
    threshold: float = 0.8,
):
    base_url = f"http://{host}:{port}"
    
    if clear:
        print(f"\n🧹 [Step 0] 清空库...")
        try: httpx.delete(f"{base_url}/api/rules"); print("   ✅ 已清空")
        except: pass

    if rules_dir:
        print(f"\n🔵 [Step 1] 建立规则库: {rules_dir}")
        rule_files = get_images_from_dir(rules_dir)
        if rule_files:
            files_payload = []
            handles = []
            for n, p in rule_files:
                f = open(p, "rb")
                handles.append(f)
                files_payload.append(("files", (n, f, "image/png")))
            try:
                print(f"   🚀 上传 {len(files_payload)} 张...")
                httpx.post(f"{base_url}/api/rules/add", files=files_payload, timeout=60.0)
                print("   ✅ 入库完成")
            finally:
                for f in handles: f.close()

    print(f"\n🔴 [Step 2] 识别测试: {query_dir}")
    query_files = get_images_from_dir(query_dir)
    if query_files:
        files_payload = []
        handles = []
        for n, p in query_files:
            f = open(p, "rb")
            handles.append(f)
            files_payload.append(("files", (n, f, "image/png")))
        try:
            print(f"   🚀 发送 {len(files_payload)} 张...")
            resp = httpx.post(f"{base_url}/api/recognize", files=files_payload, data={"top_n": top_n, "threshold": threshold}, timeout=120.0)
            if resp.status_code == 200:
                generate_html_report(resp.json(), output, query_dir)
            else:
                print(f"❌ 失败: {resp.text}")
        finally:
            for f in handles: f.close()

if __name__ == "__main__":
    app()