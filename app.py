import os
import uuid
from datetime import datetime
import logging
import traceback

from flask import Flask, render_template, request, send_file, jsonify
import gradio as gr
import pandas as pd

# Import the existing functions from your original script
from six_direction import Packer, Bin, Item, Painter, Axis
import multiprocessing
from multiprocessing import Pool, freeze_support
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import uuid
import random
from matplotlib import colors as mcolors
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Ensure results directory exists
RESULTS_DIR = os.path.join(os.getcwd(), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
def AutoFitColumnSize(worksheet, columns=None, margin=4):
    for i, column_cells in enumerate(worksheet.columns):
        is_ok = False
        if columns == None:
            is_ok = True
        elif isinstance(columns, list) and i in columns:
            is_ok = True
            
        if is_ok:
            max_length = 0
            for cell in column_cells:
                try:
                    length = len(str(cell.value))
                    if length > max_length:
                        max_length = length
                except:
                    pass
            adjusted_width = (max_length + margin)*1.1
            worksheet.column_dimensions[column_cells[0].column_letter].width = adjusted_width
    return worksheet

# 셀너비 제어 함수
def set_column_widths(writer, sheet_name, df):
    worksheet = writer.sheets[sheet_name]
    AutoFitColumnSize(worksheet)


# 색상 생성 함수
def generate_color(partno):
    random.seed(hash(partno))  # 고유한 색상을 위해 해시 사용
    hex_color = "#{:06x}".format(random.randint(0,0xFFFFF))
    color_name = get_color_name(hex_color)
    return color_name

def get_color_name(hex_color):
    try:
        closest_name = None
        min_dist = float('inf')
        r, g, b = mcolors.hex2color(hex_color)
        for name, hex in mcolors.CSS4_COLORS.items():
            r2, g2, b2 = mcolors.hex2color(hex)
            dist = (r - r2)**2 + (g - g2)**2 + (b - b2)**2
            if dist < min_dist:
                closest_name = name
                min_dist = dist
        return closest_name
    except:
        return hex_color

def read_excel(file_path):
    """
    엑셀 파일을 읽어 '투입컨테이너'와 '화물'데이터를 반환하는 함수
    """
    # 엑셀 파일 읽기
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    box_df = pd.read_excel(xls, '투입컨테이너')
    items_df = pd.read_excel(xls, '화물')

    return box_df, items_df

def parse_excel(box_df, items_df):
    """
    box_df와 items_df 데이터를 받아 Bin과 Item리스트를 생성하는 함수
    """
    # Your existing parse_excel implementation from the original script
    bins = []
    items = []
    # box 정보 처리
    for _, row in box_df.iterrows():
        if pd.notna(row['선택']) and row['선택'].strip() != '':
            WHD = (float(row['가로(mm)']), float(row['세로(mm)']), float(row['높이(mm)']))
            bins.append(Bin(
                partno=str(row['컨테이너유형']),
                WHD=WHD,
                max_weight=float(row['적재중량(MaxWeight)']),
                corner=0,
                put_type=1
            ))
    # Item 정보 처리
    for index, row in items_df.iterrows():
        WHD = (float(row['가로(mm)']),float(row['세로(mm)']),float(row['높이(mm)']))
        color = row['색깔'] if '색깔' in row and pd.notna(row['색깔']) else generate_color(row['화물명'])
        updown = str(row['회전적재']).strip() == '가능'
        for _ in range(int(row['수량'])): # 수량만큼 추가
            items.append(Item(
                partno=str(row['화물명']),
                name=str(row['화물유형']),
                typeof='cube', # 기본값 설정
                WHD=WHD,
                weight=float(row.get('중량',1)),
                level=1, # 기본값 설정
                loadbear=100, # 기본값 설정
                updown=True,
                color=color
            ))
    return bins, items
def calculate_packing(bins, items, height, depth, width):
    """
    Bin과 Item 리스트를 받아 적재 계획을 생성하는 함수
    """
    Axis.HEIGHT = height
    Axis.DEPTH = depth
    Axis.WIDTH = width

    packer = Packer()
    for b in bins:
        packer.addBin(b)
    for i in items:
        packer.addItem(i)
    
    # 적재 계획 생성
    packer.calculate(
        bigger_first=True,
        fix_point=True,
        check_stable=True,
        support_surface_ratio=1,
        number_of_decimals=0
    )
    return packer

def process_direction(height, depth, width, flag, bins, items):
    logging.info(f"Direction {flag+1} processing started......")
    packer = calculate_packing(bins, items, height, depth, width)

    volume_ratios = []
    total_weights = []

    # 부피적재율 계산
    if len(packer.bins) > 2:
        bins_without_last = packer.bins[:-1]
        for bin in bins_without_last:
            total_bin_volume = float(bin.width)*float(bin.height)*float(bin.depth)
            total_volume_used = sum(float(item.width)*float(item.height)*float(item.depth) for item in bin.items)
            volume_usage_percent = (total_volume_used/total_bin_volume)*100 if total_bin_volume > 0 else 0
            volume_ratios.append(volume_usage_percent)

            # 각 컨테이너의 총 중량 계산
            total_weight = sum(item.weight for item in bin.items)
            total_weights.append(total_weight)

            # 상세 정보 출력
            logging.info(f"Container {bin.partno}:")
            logging.info(f"  - Volume ratio: {volume_usage_percent:.2f}%")
            logging.info(f"  - Total weight: {total_weight:.2f}kg")
        average_volume_usage = sum(volume_ratios) / len(volume_ratios) if volume_ratios else 0
    else:
        first_bin = packer.bins[0]
        total_bin_volume = float(first_bin.width)*float(first_bin.height)*float(first_bin.depth)
        total_volume_used = sum(float(item.width)*float(item.height)*float(item.depth) for item in first_bin.items)
        average_volume_usage = (total_volume_used / total_bin_volume) * 100 if total_bin_volume>0 else 0

        for bin in packer.bins:
            total_bin_volume = float(bin.width)*float(bin.height)*float(bin.depth)
            total_volume_used = sum(float(item.width)*float(item.height)*float(item.depth) for item in bin.items)
            volume_usage_percent = (total_volume_used/total_bin_volume)*100 if total_bin_volume > 0 else 0
            volume_ratios.append(volume_usage_percent)

            total_weight = sum(item.weight for item in bin.items)
            total_weights.append(total_weight)
            # 상세 정보 출력
            logging.info(f"Container {bin.partno}:")
            logging.info(f"  - Volume ratio: {volume_usage_percent:.2f}%")
            logging.info(f"  - Total weight: {total_weight:.2f}kg")
    
    logging.info(f"Direction {flag+1} processing completed with volume usage: {average_volume_usage}")
    
    #result_dict[flag] = (average_volume_usage, packer)
    return flag, average_volume_usage, packer, volume_ratios, total_weights

def find_optimal_direction(result_dict):
    """
    최적의 적재 방향을 찾는 함수
    """
    # 부피 적재율이 가장 높은 결과 선택
    optimal_flag = max(result_dict, key=lambda flag: result_dict[flag][0])  # volume 값을 기준으로 최대값 찾기
    optimal_volume, optimal_packer = result_dict[optimal_flag]
    return optimal_flag, optimal_volume, optimal_packer

def start_parallel_processing_with_pool(bins, items):
    prob = {'HEIGHT': [0, 0, 1, 2, 1, 2], 'DEPTH': [1, 2, 2, 1, 0, 0], 'WIDTH': [2, 1, 0, 0, 2, 1]}
    
    with Pool(processes=6) as pool:
        results = pool.starmap(process_direction, [(prob['HEIGHT'][i], prob['DEPTH'][i], prob['WIDTH'][i], i, bins, items) for i in range(6)])

    result_dict = {}
    for result in results:
        flag, volume, packer, volume_ratios, total_weights = result
        result_dict[flag] = (volume, packer, volume_ratios, total_weights)
    
    optimal_flag = max(result_dict, key=lambda x: result_dict[x][0])
    optimal_volume = result_dict[optimal_flag][0]
    optimal_packer = result_dict[optimal_flag][1]
    
    return optimal_flag, optimal_volume, optimal_packer, result_dict
def run_process_direction(height, depth, width, flag, bins, items):
    # cytohon 모듈은 여기서 직접 호출하도록 수정
    return process_direction.run(height, depth, width, flag, bins, items)

def save_packing_results(bins_df, items_df, optimal_packer, result_excel_path):
    """
    적재 계획 결과를 엑셀 파일로 저장하는 함수
    """
    results_df, summary_df, additional_info_df, additional_info_2_df = summary_result(optimal_packer)

    with pd.ExcelWriter(result_excel_path) as writer:
        # input data도 함께 저장
        bins_df.to_excel(writer, sheet_name="투입컨테이너", index=False)
        items_df.to_excel(writer, sheet_name='화물', index=False)
        results_df.to_excel(writer, sheet_name='화물적재상세계획', index=False)
        summary_df.to_excel(writer, sheet_name='컨테이너별적재결과',index=False)

        # 미적재화물 시트 저장
        additional_info_df.to_excel(writer, sheet_name='미적재화물', index=False, startrow=0)
        additional_info_2_df.to_excel(writer, sheet_name='미적재화물', index=False, startrow=2)

        # 셀 크기 조절
        set_column_widths(writer, '투입컨테이너', bins_df)
        set_column_widths(writer, '화물', items_df)
        set_column_widths(writer, '화물적재상세계획', results_df)
        set_column_widths(writer, '컨테이너별적재결과', summary_df)
        set_column_widths(writer, '미적재화물', additional_info_df)

# 그림 그리기 함수
def draw_packing_results(bins, spath, step_interval=5, use_grouped=False):
    image_paths = []

    # 다양한 각도에서 보기 위한 각도 설정
    view_angles = [
        (20, -60),   # Top-Front view
        # (90, 0),     # Side view
        # (0, 90),     # Front view
        # (30, 30)     # Isometric view
    ]
    
    # 1. 전체 적재 상태 이미지 생성
    for i, b in enumerate(bins):
        if len(b.items) > 0:  # 아이템이 있는지 확인
            painter = Painter(b)
            
            # 전체 모습(최종 적재 상태) 이미지 생성
            fig = painter.plotBoxAndItems(
                title=f"{b.partno} - Full View - {i+1}",
                alpha=0.8,
                write_num=False,
                fontsize=10
            )

        # fig 객체 확인 후 이미지 저장 로직 실행
        if isinstance(fig, plt.Figure):  # fig가 올바른 Figure 객체인지 확인
            axGlob = fig.gca()  # 현재 Axes 객체를 가져옴
            
            # 축의 비율을 맞추기 위해 set_axes_equal 호출
            painter.setAxesEqual(axGlob)
            # 다양한 각도에서 이미지 저장
            for angle_idx, (elev, azim) in enumerate(view_angles):
                axGlob.view_init(elev=elev, azim=azim)
                final_image_path = spath + f"{b.partno}_final_plot_{i}_view_{angle_idx}.png"
                fig.savefig(final_image_path)
                image_paths.append(final_image_path)

            plt.close(fig)
        else:
            logging.info(f"an unexpected objects: {type(fig)}")

    # 2. 단계별 쌓이는 과정 이미지 생성
    for i, b in enumerate(bins):
        if len(b.items) > 0:  # 아이템이 있는지 확인
            painter = Painter(b)

            if use_grouped:
                # 그룹별로 화물이 적재되는 이미지 생성
                step_images = painter.plotBoxAndItemsStepByStep(
                    title=f"{b.partno} - Step View (Grouped) - {i+1}",
                    alpha=0.8,
                    write_num=False,
                    fontsize=10
                )
            else:
                # step_interval에 따른 화물이 적재되는 이미지 생성
                # 화물을 단계적으로 적재하는 과정 이미지 생성
                step_images = painter.plotBoxAndItemsByInterval(
                    title=f"{b.partno} - Step View (Interval) - {i+1}",
                    alpha=0.8,
                    write_num=False,
                    fontsize=10,
                    step_interval=step_interval
                )

            # 각 단계별 이미지 저장
            for step_idx, step_fig in enumerate(step_images):
                axGlob = step_fig.gca()  # 현재 Axes 객체를 가져옴
                axGlob.set_title(f"{b.partno} - Step View - {step_idx + 1}", fontsize=18)

                painter.setAxesEqual(axGlob)
                # 다양한 각도에서 이미지 저장
                for angle_idx, (elev, azim) in enumerate(view_angles):
                    axGlob.view_init(elev=elev, azim=azim)
                    step_image_path = f"{spath}{b.partno}_step_{step_idx + 1}_view_{angle_idx+1}.png"
                    step_fig.savefig(step_image_path)
                    image_paths.append(step_image_path)

                plt.close(step_fig)

    return image_paths

def summary_result(packer):
    results = []
    summary = []
    volume_t = 0
    volume_f = 0
    unfitted_items = defaultdict(int)

    for index, b in enumerate(packer.bins) :
        volume = b.width * b.height * b.depth
        volume_t = 0
        order = 1
        for item in b.items:
            results.append({
                "컨테이너유형" : b.partno,
                "화물명" : item.partno,
                "화물유형" : item.name,
                "색깔" : item.color,
                "위치" : [float(pos) for pos in item.position],
                "가로(mm)" : item.width,
                "세로(mm)" : item.height,
                "높이(mm)" : item.depth,
                "부피(mm3)" : f"{int(float(item.width) * float(item.height) * float(item.depth)):,}",  # 부피에 천 단위 콤마 추가
                "중량" : float(item.weight),
                "순서" : order
            })
            volume_t += float(item.width) * float(item.height) * float(item.depth)
            order += 1
        space_utilization = round(volume_t / float(volume) * 100, 2)
        residual_volume = f"{int(float(volume) - volume_t):,}"
        gravity_distribution = b.gravity
        num_items = len(b.items)
        
        summary.append({
            "컨테이너번호": b.partno,
            "적재화물수" : num_items,
            "공간활용도 (%)": space_utilization,
            "잔여공간": residual_volume
        })

        if index < len(packer.bins) -1:
            results.append({})

    results.append({})
    for item in packer.unfit_items:
        results.append({
            "컨테이너유형": "unfitted",
            "화물명" : item.partno,
            "화물유형" : item.name,
            "색깔": item.color,
            "위치": "unfitted",
            "가로(mm)" : item.width,
            "세로(mm)" : item.height,
            "높이(mm)" : item.depth,
            "부피(mm3)": f"{int(float(item.width) * float(item.height) * float(item.depth)):,}",
            "중량": float(item.weight)
        })
        partno = item.partno
        unfitted_items[partno] += 1
        volume_f += float(item.width) * float(item.height) * float(item.depth)

    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary)

    additional_info_df = pd.DataFrame({
        "미적재화물 부피": [f"{int(volume_f):,}"]
    })

    additional_info_2_df = pd.DataFrame({
        "미적재화물명": list(unfitted_items.keys()),
        "수량": list(unfitted_items.values())
            })
    return results_df, summary_df, additional_info_df, additional_info_2_df

def save_and_generate(bins_df, items_df, optimal_packer, result_excel_path, spath):
    with ThreadPoolExecutor() as executor:
        # 파일 저장과 이미지 생성을 병렬로 실행
        future_excel = executor.submit(save_packing_results, bins_df, items_df, optimal_packer, result_excel_path)
        future_images = executor.submit(draw_packing_results, optimal_packer.bins, spath, use_grouped=False)

        # 두 작업의 결과를 가져옴
        future_excel.result() # 파일 저장 결과 (필요시 활용)
        image_paths = future_images.result() # 이미지 경로 리스트 반환

    return result_excel_path, image_paths

# Include other necessary functions from your original script:
# process_direction, generate_color, summary_result, save_packing_results, draw_packing_results, save_and_generate

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the uploaded file temporarily
        now = datetime.now()
        formatted_time = now.strftime("%Y%m%d-%H%M%S-%f")
        temp_filename = os.path.join(RESULTS_DIR, f"temp_{formatted_time}.xlsx")
        file.save(temp_filename)

        # Process the file
        box_df, items_df = read_excel(temp_filename)
        bins, items = parse_excel(box_df, items_df)

        # Perform optimization
        optimal_flag, optimal_volume, optimal_packer, result_dict = start_parallel_processing_with_pool(bins, items)

        # Prepare result file path
        result_excel_path = os.path.join(RESULTS_DIR, f"적재계획_{formatted_time}.xlsx")
        
        # Generate result file and images
        result_excel_path, image_paths = save_and_generate(box_df, items_df, optimal_packer, result_excel_path, RESULTS_DIR + '/')

        # Prepare result summary
        result_summary = f"최적 방향: {optimal_flag + 1}, 부피 적재율: {optimal_volume:.2f}%\n"
        for flag, (volume, _, volume_ratios, total_weights) in result_dict.items():
            result_summary += f"\nDirection {flag + 1}:\n"
            for idx, (volume_ratio, weight) in enumerate(zip(volume_ratios, total_weights)):
                result_summary += f"  컨테이너 {idx + 1}: 적재율 {volume_ratio:.2f}%, 총중량 {weight:.2f}kg\n"

        return jsonify({
            "excel_path": result_excel_path,
            "image_paths": image_paths,
            "summary": result_summary
        })

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return str(e), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)