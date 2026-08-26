from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

async def read_excel(filename: str, sheet_name: str) -> str:
    """
    读取Excel指定工作表；
    当传入的sheet_name不存在时，会直接返回该文件全部工作表名称，方便重新选择。
    """
    if not filename or not sheet_name or len(filename) > 255 or len(sheet_name) > 128:
        return "错误：文件名和工作表名不能为空且长度超限。"

    data_root = DATA_DIR.resolve()
    filepath = (data_root / filename).resolve()
    try:
        filepath.relative_to(data_root)
    except ValueError:
        return "错误：只能读取 data 目录中的文件。"
    if filepath.suffix.lower() not in {".xlsx", ".xls"}:
        return "错误：只支持 .xlsx 和 .xls 文件。"
    if not filepath.exists():
        return f"错误：文件 '{filename}' 不存在。"

    try:
        excel_file = pd.ExcelFile(filepath)
        sheet_list = excel_file.sheet_names

        if sheet_name not in sheet_list:
            # 指定工作表不存在，直接返回全部sheet，交给大模型重试
            return f"错误：工作表【{sheet_name}】不存在！该文件可用工作表列表：{sheet_list}"

        df = pd.read_excel(filepath, sheet_name=sheet_name)
        rows_total = len(df)
        cols_total = len(df.columns)
        preview_df = df.head(20)
        preview_text = preview_df.to_string()
        out = f"【{filename}】工作表:{sheet_name}，共{rows_total}行 {cols_total}列\n{preview_text}"
        if rows_total > 20:
            out += f"\n⚠️仅展示前20行，完整数据共{rows_total}行"
        return out

    except Exception as e:
        return f"读取Excel失败：{str(e)}"