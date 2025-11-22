import typer
import httpx
import os
from pathlib import Path
from typing import Optional, List, Tuple
from typing_extensions import Annotated
import json

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
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f0f2f5; color: #333; }
        .container { max-width: 1300px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        h1 { border-bottom: 2px solid #eee; padding-bottom: 15px; color: #1a237e; margin-top: 0; }
        .stats { background: #e8eaf6; padding: 15px; border-radius: 8px; color: #3f51b5; font-weight: 500; margin-bottom: 30px; }
        .global-viz { text-align: center; margin: 30px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 25px; }
        .card { border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; background: #fff; display: flex; flex-direction: column; }
        .card-header { background: #f5f5f5; padding: 10px 15px; font-weight: bold; display: flex; justify-content: space-between; }
        
        .match-list { max-height: 250px; overflow-y: auto; border-bottom: 1px solid #eee; }
        .match-item { 
            display: grid; grid-template-columns: 30px 1fr 160px 70px; 
            align-items: center; padding: 8px 12px; border-bottom: 1px solid #f0f0f0; 
            cursor: pointer; font-size: 0.9em;
        }
        .match-item:hover { background: #e3f2fd; }
        .match-item.active { background: #bbdefb; border-left: 4px solid #2196f3; }
        
        .rank-badge { background: #9e9e9e; color: white; width: 20px; height: 20px; border-radius: 50%; text-align: center; font-size: 11px; line-height: 20px; }
        .rank-1 { background: #ff9800; }
        
        .viz-container { padding: 10px; background: #222; text-align: center; min-height: 200px; display: flex; justify-content: center; align-items: center; }
        .viz-img { max-width: 100%; border: 1px solid #444; }
        
        .tag { padding: 2px 6px; border-radius: 4px; font-size: 0.7em; font-weight: bold; text-align: center; }
        .tag-trusted { background: #e8f5e9; color: #2e7d32; }
        .tag-untrusted { background: #ffebee; color: #c62828; }
        
        .score-val { font-family: monospace; font-weight: bold; color: #1565c0; font-size: 1.1em; }
        .detail-stats { font-size: 0.75em; color: #666; display: flex; flex-direction: column; align-items: flex-end; }
        .stat-row { display: flex; gap: 5px; }
        .stat-key { color: #999; }
    </style>
    
    <script>
        const imgData = {};
        function showViz(cardId, rankIdx, btn) {
            const imgId = 'img-' + cardId;
            const imgEl = document.getElementById(imgId);
            const dataKey = cardId + '-' + rankIdx;
            if (imgData[dataKey]) { imgEl.src = "data:image/png;base64," + imgData[dataKey]; }
            const listId = 'list-' + cardId;
            const listEl = document.getElementById(listId);
            for (let item of listEl.getElementsByClassName('match-item')) { item.classList.remove('active'); }
            btn.classList.add('active');
        }
    </script>
    """

    html_parts = []
    html_parts.append(f"<!DOCTYPE html><html><head><title>CircuitSight Report</title>{css_style}</head><body><div class='container'><h1>CircuitSight 识别测试报告</h1>")
    html_parts.append(f"<div class='stats'>📂 测试目录: {query_dir} &nbsp;|&nbsp; 🖼️ 图片数量: {total}</div>")

    if data.get("global_visualization"):
        html_parts.append(f"<div class='global-viz'><h3>全局向量分布 (PCA)</h3><img src='data:image/png;base64,{data['global_visualization']}' style='max-width:800px'/></div>")

    html_parts.append("<div class='grid'>")
    
    js_data_injection = []

    for file_idx, item in enumerate(data["results"]):
        card_id = f"card-{file_idx}"
        matches_html = f"<div class='match-list' id='list-{card_id}'>"
        default_viz = ""
        best_trusted = False
        
        for i, m in enumerate(item["matches"]):
            is_trusted = m.get('trusted', False)
            score = m.get('score', 0.0)
            
            # 获取网格统计
            kp_q = m.get('kp_query_total', 0) # Matched Grids
            kp_r = m.get('kp_ref_total', 0)   # Total Max Grids
            
            viz_b64 = m.get('viz_base64')
            if viz_b64:
                js_data_injection.append(f"imgData['{card_id}-{i}'] = '{viz_b64}';")
                if i == 0: default_viz = viz_b64

            rank_cls = "rank-1" if i == 0 else ""
            status_cls = "tag-trusted" if is_trusted else "tag-untrusted"
            status_text = "OK" if is_trusted else "Low"
            active_cls = "active" if i == 0 else ""
            if i == 0 and is_trusted: best_trusted = True
            
            matches_html += f"""
            <div class="match-item {active_cls}" onclick="showViz('{card_id}', {i}, this)">
                <div class="rank-badge {rank_cls}">{i+1}</div>
                <div style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{m['label']}">{m['label']}</div>
                <div class="detail-stats">
                    <div class="stat-row">
                        <span class="score-val">{score:.1f}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">Grid Match:</span> {kp_q}/{kp_r}
                    </div>
                </div>
                <div class="tag {status_cls}">{status_text}</div>
            </div>"""
        matches_html += "</div>"

        icon = "✅" if best_trusted else "⚠️"
        img_src = f"data:image/png;base64,{default_viz}" if default_viz else ""
        
        html_parts.append(f"""
        <div class="card">
            <div class="card-header"><span title="{item['filename']}">{icon} {item['filename']}</span></div>
            <div class="card-body">
                {matches_html}
                <div class="viz-container">
                    <img id="img-{card_id}" class="viz-img" src="{img_src}" />
                </div>
            </div>
        </div>
        """)

    html_parts.append("<script>" + "".join(js_data_injection) + "</script>")
    html_parts.append("</div></div></body></html>")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f: f.write("".join(html_parts))
    print(f"\n✅ 报告已生成: {output_file.absolute()}")


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
        try: httpx.delete(f"{base_url}/api/rules", timeout=10.0)
        except: pass
    if rules_dir:
        files = get_images_from_dir(rules_dir)
        if files:
            payload = [("files", (n, open(p, "rb"), "image/png")) for n, p in files]
            try: httpx.post(f"{base_url}/api/rules/add", files=payload, timeout=60.0)
            finally: 
                for _, (_, f, _) in payload: f.close()
    
    q_files = get_images_from_dir(query_dir)
    if q_files:
        payload = [("files", (n, open(p, "rb"), "image/png")) for n, p in q_files]
        try:
            resp = httpx.post(f"{base_url}/api/recognize", files=payload, data={"top_n": top_n, "threshold": threshold}, timeout=120.0)
            if resp.status_code == 200: generate_html_report(resp.json(), output, query_dir)
        finally:
            for _, (_, f, _) in payload: f.close()

if __name__ == "__main__":
    app()