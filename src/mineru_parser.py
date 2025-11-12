# src/mineru_parser.py

import requests
import logging
import time
import json
import zipfile
import io
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

_log = logging.getLogger(__name__)


class MineruJsonReportProcessor:
    """处理 Mineru API 返回的 JSON 数据，转换为标准格式"""
    
    def __init__(self, metadata_lookup: dict = None):
        self.metadata_lookup = metadata_lookup or {}
    
    def assemble_report(self, content_list: List[Dict], pdf_path: Path = None) -> Dict[str, Any]:
        """
        将 Mineru 的 content_list 转换为标准报告格式
        
        Args:
            content_list: Mineru API 返回的内容列表
            pdf_path: PDF 文件路径（用于提取 sha1 名称）
        
        Returns:
            格式化的报告字典
        """
        assembled_report = {}
        
        # 组装元信息
        assembled_report['metainfo'] = self._assemble_metainfo(content_list, pdf_path)
        
        # 组装内容
        assembled_report['content'] = self._assemble_content(content_list)
        
        # 组装表格
        assembled_report['tables'] = self._assemble_tables(content_list)
        
        # 组装图片
        assembled_report['pictures'] = self._assemble_pictures(content_list)
        
        return assembled_report
    
    def _assemble_metainfo(self, content_list: List[Dict], pdf_path: Path = None) -> Dict[str, Any]:
        """组装元信息"""
        metainfo = {}
        
        # 从 PDF 路径提取 sha1 名称
        if pdf_path:
            sha1_name = pdf_path.stem
            metainfo['sha1_name'] = sha1_name
            
            # 添加 CSV 元数据（如果有）
            if self.metadata_lookup and sha1_name in self.metadata_lookup:
                csv_meta = self.metadata_lookup[sha1_name]
                metainfo['company_name'] = csv_meta.get('company_name', '')
        else:
            metainfo['sha1_name'] = 'unknown'
        
        # 统计各类型内容的数量
        pages = set()
        text_blocks = 0
        tables = 0
        pictures = 0
        equations = 0
        footnotes = 0
        
        for item in content_list:
            page_idx = item.get('page_idx', 0)
            pages.add(page_idx)
            
            item_type = item.get('type', '')
            
            if item_type == 'text':
                text_blocks += 1
            elif item_type == 'table':
                tables += 1
            elif item_type == 'image':
                pictures += 1
            elif item_type == 'equation':
                equations += 1
            elif item_type == 'footer':
                footnotes += 1
        
        metainfo['pages_amount'] = len(pages) if pages else 0
        metainfo['text_blocks_amount'] = text_blocks
        metainfo['tables_amount'] = tables
        metainfo['pictures_amount'] = pictures
        metainfo['equations_amount'] = equations
        metainfo['footnotes_amount'] = footnotes
        
        return metainfo
    
    def _assemble_content(self, content_list: List[Dict]) -> List[Dict[str, Any]]:
        """按页组织内容"""
        pages_dict = defaultdict(lambda: {'content': []})
        
        for idx, item in enumerate(content_list):
            page_num = item.get('page_idx', 0) + 1  # 转换为 1-based 索引
            item_type = item.get('type', '')
            
            content_item = {}
            
            if item_type == 'text':
                content_item = {
                    'text': item.get('text', ''),
                    'type': item_type,
                    'text_id': idx
                }
                # 添加文本级别（如果有）
                if 'text_level' in item:
                    content_item['text_level'] = item['text_level']
                    
            elif item_type == 'list':
                # 列表类型转换为文本
                list_items = item.get('list_items', [])
                list_text = '\n'.join(list_items)
                content_item = {
                    'text': list_text,
                    'type': 'text',
                    'text_id': idx
                }
                
            elif item_type == 'table':
                content_item = {
                    'type': 'table',
                    'table_id': idx
                }
                
            elif item_type == 'image':
                content_item = {
                    'type': 'picture',
                    'picture_id': idx
                }
                
            elif item_type == 'footer':
                content_item = {
                    'text': item.get('text', ''),
                    'type': 'footnote',
                    'text_id': idx
                }
                
            elif item_type == 'header':
                content_item = {
                    'text': item.get('text', ''),
                    'type': 'page-header',
                    'text_id': idx
                }
            else:
                # 其他类型也作为文本处理
                content_item = {
                    'text': item.get('text', ''),
                    'type': item_type,
                    'text_id': idx
                }
            
            # 添加到对应页面
            if content_item:
                pages_dict[page_num]['content'].append(content_item)
                
                # 保存页面维度信息
                if 'bbox' in item and 'page_dimensions' not in pages_dict[page_num]:
                    pages_dict[page_num]['page_dimensions'] = item.get('bbox', {})
        
        # 转换为排序后的列表
        sorted_pages = []
        for page_num in sorted(pages_dict.keys()):
            page_data = pages_dict[page_num]
            page_data['page'] = page_num
            sorted_pages.append(page_data)
        
        return sorted_pages
    
    def _assemble_tables(self, content_list: List[Dict]) -> List[Dict[str, Any]]:
        """组装表格信息"""
        assembled_tables = []
        
        for idx, item in enumerate(content_list):
            if item.get('type') != 'table':
                continue
            
            page_num = item.get('page_idx', 0) + 1  # 转换为 1-based 索引
            bbox = item.get('bbox', [0, 0, 0, 0])
            
            # 提取表格 HTML
            table_html = item.get('table_body', '')
            
            # 将 HTML 转换为 Markdown
            table_md = self._html_to_markdown(table_html)
            
            # 计算行列数
            nrows, ncols = self._count_table_dimensions(table_html)
            
            table_obj = {
                'table_id': idx,
                'page': page_num,
                'bbox': bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'caption': item.get('table_caption', []),
                'footnote': item.get('table_footnote', [])
            }
            
            # 添加图片路径（如果有）
            if 'img_path' in item:
                table_obj['img_path'] = item['img_path']
            
            assembled_tables.append(table_obj)
        
        return assembled_tables
    
    def _assemble_pictures(self, content_list: List[Dict]) -> List[Dict[str, Any]]:
        """组装图片信息"""
        assembled_pictures = []
        
        for idx, item in enumerate(content_list):
            if item.get('type') != 'image':
                continue
            
            page_num = item.get('page_idx', 0) + 1  # 转换为 1-based 索引
            bbox = item.get('bbox', [0, 0, 0, 0])
            
            picture_obj = {
                'picture_id': idx,
                'page': page_num,
                'bbox': bbox,
                'img_path': item.get('img_path', '')
            }
            
            # 添加图片标题和脚注（如果有）
            if 'image_caption' in item:
                picture_obj['caption'] = item['image_caption']
            if 'image_footnote' in item:
                picture_obj['footnote'] = item['image_footnote']
            
            assembled_pictures.append(picture_obj)
        
        return assembled_pictures
    
    def _html_to_markdown(self, html: str) -> str:
        """将 HTML 表格转换为 Markdown 格式"""
        if not html:
            return ''
        
        try:
            # 简单的 HTML 表格解析
            rows = []
            
            # 提取所有 <tr> 标签
            tr_pattern = r'<tr>(.*?)</tr>'
            tr_matches = re.findall(tr_pattern, html, re.DOTALL)
            
            for tr_match in tr_matches:
                # 提取所有 <td> 或 <th> 标签
                td_pattern = r'<t[dh]>(.*?)</t[dh]>'
                cells = re.findall(td_pattern, tr_match, re.DOTALL)
                # 清理单元格内容
                cells = [cell.strip() for cell in cells]
                rows.append(cells)
            
            if not rows:
                return html
            
            # 构建 Markdown 表格
            md_lines = []
            
            # 表头
            if len(rows) > 0:
                header = ' | '.join(rows[0])
                md_lines.append(f"| {header} |")
                
                # 分隔符
                separator = ' | '.join(['---'] * len(rows[0]))
                md_lines.append(f"| {separator} |")
                
                # 表体
                for row in rows[1:]:
                    # 确保列数一致
                    while len(row) < len(rows[0]):
                        row.append('')
                    row_text = ' | '.join(row[:len(rows[0])])
                    md_lines.append(f"| {row_text} |")
            
            return '\n'.join(md_lines)
            
        except Exception as e:
            _log.warning(f"转换 HTML 表格为 Markdown 时出错: {e}")
            return html
    
    def _count_table_dimensions(self, html: str) -> tuple:
        """统计表格的行数和列数"""
        if not html:
            return (0, 0)
        
        try:
            # 提取所有 <tr> 标签
            tr_pattern = r'<tr>(.*?)</tr>'
            tr_matches = re.findall(tr_pattern, html, re.DOTALL)
            nrows = len(tr_matches)
            
            # 从第一行提取列数
            ncols = 0
            if tr_matches:
                td_pattern = r'<t[dh]>.*?</t[dh]>'
                cells = re.findall(td_pattern, tr_matches[0], re.DOTALL)
                ncols = len(cells)
            
            return (nrows, ncols)
            
        except Exception as e:
            _log.warning(f"统计表格维度时出错: {e}")
            return (0, 0)


class MineruPDFParser:
    def __init__(
        self, 
        api_token: str, 
        output_dir: Path, 
        debug_data_path: Path = None,
        csv_metadata_path: Path = None
    ):
        self.api_token = api_token
        self.output_dir = output_dir
        self.debug_data_path = debug_data_path
        self.batch_url = "https://mineru.net/api/v4/file-urls/batch"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_data_path:
            self.debug_data_path.mkdir(parents=True, exist_ok=True)
        
        # 解析 CSV 元数据
        self.metadata_lookup = {}
        if csv_metadata_path is not None:
            self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path)
    
    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """解析 CSV 文件并创建查找字典，以 sha1 为键"""
        import csv
        metadata_lookup = {}
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 兼容新旧 CSV 格式中的公司名称字段
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name
                }
        return metadata_lookup

    def _create_batch_and_upload(self, pdf_paths: List[Path]) -> str:
        """创建批次并上传 PDF 文件，返回 batch_id"""
        # 准备文件列表
        files = []
        for idx, pdf_path in enumerate(pdf_paths):
            files.append({
                "name": str(pdf_path),
                "data_id": f"file_{idx}"
            })
        
        data = {
            "files": files,
            "model_version": "vlm"
        }
        
        try:
            # 第一步：申请上传 URL
            response = requests.post(self.batch_url, headers=self.headers, json=data)
            if response.status_code != 200:
                _log.error(f"申请上传 URL 失败，状态码: {response.status_code}")
                return None
            
            result = response.json()
            if result.get("code") != 0:
                _log.error(f"申请上传 URL 失败: {result.get('msg')}")
                return None
            
            batch_id = result["data"]["batch_id"]
            file_urls = result["data"]["file_urls"]
            _log.info(f"成功创建批次，batch_id: {batch_id}")
            
            # 第二步：上传文件
            for i, upload_url in enumerate(file_urls):
                pdf_path = pdf_paths[i]
                _log.info(f"正在上传文件: {pdf_path}")
                
                with open(pdf_path, 'rb') as f:
                    upload_response = requests.put(upload_url, data=f)
                    if upload_response.status_code == 200:
                        _log.info(f"文件 {pdf_path.name} 上传成功")
                    else:
                        _log.error(f"文件 {pdf_path.name} 上传失败，状态码: {upload_response.status_code}")
                        return None
            
            return batch_id
            
        except requests.exceptions.RequestException as e:
            _log.error(f"创建批次或上传文件失败: {e}")
            return None

    def _get_extract_results(self, batch_id: str, max_retries: int = 60, retry_interval: int = 10) -> List[Dict[str, Any]]:
        """获取提取结果，返回提取结果列表"""
        extract_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(extract_url, headers=self.headers)
                if response.status_code != 200:
                    _log.warning(f"获取提取结果失败，状态码: {response.status_code}，重试中...")
                    time.sleep(retry_interval)
                    continue
                
                result = response.json()
                if result.get("code") != 0:
                    _log.warning(f"获取提取结果失败: {result.get('msg')}，重试中...")
                    time.sleep(retry_interval)
                    continue
                
                extract_results = result["data"]["extract_result"]
                
                # 检查所有文件是否都已完成
                all_done = all(item["state"] == "done" for item in extract_results)
                if not all_done:
                    _log.info(f"批次处理中，等待 {retry_interval} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_interval)
                    continue
                
                _log.info(f"批次处理完成，共 {len(extract_results)} 个文件")
                return extract_results
                
            except requests.exceptions.RequestException as e:
                _log.error(f"获取提取结果出错: {e}")
                time.sleep(retry_interval)
        
        _log.error(f"获取提取结果超时，已重试 {max_retries} 次")
        return []

    def _download_and_extract_zip(self, zip_url: str, file_name: str, pdf_path: Path = None) -> bool:
        """下载并解压 ZIP 文件，然后处理格式化"""
        try:
            _log.info(f"正在下载结果文件: {file_name}")
            response = requests.get(zip_url)
            if response.status_code != 200:
                _log.error(f"下载失败，状态码: {response.status_code}")
                return False
            
            # 创建临时目录解压
            temp_extract_dir = self.output_dir / "temp_extract"
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            
            # 解压 ZIP 文件到临时目录
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extractall(temp_extract_dir)
                _log.info(f"文件 {file_name} 解压完成")
                
                # 查找 content_list.json 文件
                content_list_files = list(temp_extract_dir.glob("**/*_content_list.json"))
                
                if not content_list_files:
                    _log.warning(f"未找到 content_list.json 文件: {file_name}")
                    return False
                
                # 处理每个 content_list.json 文件
                for content_list_path in content_list_files:
                    self._process_content_list(content_list_path, pdf_path)
                
                # 如果需要保留原始文件，可以复制到 debug_data_path
                if self.debug_data_path:
                    # 复制整个解压目录到 debug_data_path
                    import shutil
                    debug_extract_dir = self.debug_data_path / temp_extract_dir.name
                    if debug_extract_dir.exists():
                        shutil.rmtree(debug_extract_dir)
                    shutil.copytree(temp_extract_dir, debug_extract_dir)
                
                # 清理临时目录
                import shutil
                shutil.rmtree(temp_extract_dir)
            
            return True
            
        except Exception as e:
            _log.error(f"下载或解压文件失败: {e}")
            return False
    
    def _process_content_list(self, content_list_path: Path, pdf_path: Path = None):
        """处理 content_list.json 文件并生成格式化输出"""
        try:
            # 读取 content_list.json
            with open(content_list_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            # 创建处理器
            processor = MineruJsonReportProcessor(metadata_lookup=self.metadata_lookup)
            
            # 组装报告
            assembled_report = processor.assemble_report(content_list, pdf_path)
            
            # 确定输出文件名
            if pdf_path:
                output_filename = pdf_path.stem + ".json"
            else:
                # 从 content_list 文件名提取
                base_name = content_list_path.stem.replace('_content_list', '')
                output_filename = base_name + ".json"
            
            # 保存格式化的报告
            output_path = self.output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(assembled_report, f, indent=2, ensure_ascii=False)
            
            _log.info(f"已保存格式化报告: {output_path}")
            
        except Exception as e:
            _log.error(f"处理 content_list 文件失败: {e}")
            raise

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        """
        解析 PDF 文件并导出结果。
        
        Args:
            input_doc_paths: PDF 文件路径列表
            doc_dir: PDF 文件所在目录（如果未提供 input_doc_paths）
        """
        # 获取 PDF 文件列表
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))
        elif input_doc_paths is None:
            _log.error("未提供 PDF 路径或目录")
            return
        
        if not input_doc_paths:
            _log.warning("没有找到 PDF 文件")
            return
        
        total_docs = len(input_doc_paths)
        _log.info(f"开始使用 Mineru API 处理 {total_docs} 份文档")
        
        # 第一步：创建批次并上传文件
        batch_id = self._create_batch_and_upload(input_doc_paths)
        if not batch_id:
            _log.error("创建批次或上传文件失败")
            return
        
        # 保存 batch_id 用于调试
        if self.debug_data_path:
            batch_info_path = self.debug_data_path / f"batch_{batch_id}.json"
            with open(batch_info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_id": batch_id,
                    "files": [str(p) for p in input_doc_paths],
                    "timestamp": time.time()
                }, f, indent=2, ensure_ascii=False)
        
        # 第二步：获取提取结果
        extract_results = self._get_extract_results(batch_id)
        if not extract_results:
            _log.error("获取提取结果失败")
            return
        
        # 保存原始提取结果用于调试
        if self.debug_data_path:
            results_path = self.debug_data_path / f"extract_results_{batch_id}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(extract_results, f, indent=2, ensure_ascii=False)
        
        # 第三步：下载并解压结果
        success_count = 0
        failure_count = 0
        
        # 创建文件名到 PDF 路径的映射
        pdf_path_map = {str(pdf_path): pdf_path for pdf_path in input_doc_paths}
        
        for result_item in extract_results:
            file_name = result_item.get("file_name", "unknown")
            if result_item.get("state") == "done" and result_item.get("full_zip_url"):
                zip_url = result_item["full_zip_url"]
                
                # 查找对应的 PDF 路径
                pdf_path = pdf_path_map.get(file_name)
                
                if self._download_and_extract_zip(zip_url, file_name, pdf_path):
                    success_count += 1
                else:
                    failure_count += 1
            else:
                _log.error(f"文件 {file_name} 处理失败: {result_item.get('err_msg', '未知错误')}")
                failure_count += 1
        
        _log.info(f"完成处理 {total_docs} 份文档。成功: {success_count}, 失败: {failure_count}")