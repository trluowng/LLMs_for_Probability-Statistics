import pandas as pd
import google.generativeai as genai
import json
import time

# 1. Cấu hình API Key
genai.configure(api_key='API_Key')

# 2. Khởi tạo mô hình Gemini
model = genai.GenerativeModel('gemini-3-flash-preview')

# Hàm gọi LLM để chấm điểm từng dòng
def evaluate_with_llm(question, model_answer):
    prompt = f"""
    Bạn là một giám khảo chấm điểm toán học chuyên nghiệp. 
    Dưới đây là câu hỏi và câu trả lời từng bước của mô hình:
    
    Câu hỏi: {question}
    Câu trả lời: {model_answer}
    
    Hãy chấm điểm theo thang điểm từ 0 đến 1 (trừ 1-2 điểm trên 1 bước sai) và theo các tiêu chí sau:
    1. Logic_Score: Các bước lập luận có hợp lý, liên kết chặt chẽ và không nhảy cóc không?
    2. Math_Score: Công thức và kết quả tính toán có chính xác không?
    3. Average_Score: Điểm trung bình cộng của Logic và Math.
    4. Feedback: Nhận xét ngắn gọn về lỗi sai (nếu có) hoặc khen ngợi nếu làm tốt.
    
    BẮT BUỘC trả về ĐÚNG định dạng JSON sau (không chứa bất kỳ văn bản nào khác, không dùng markdown block):
    {{
        "Logic_Score": 8,
        "Math_Score": 9,
        "Average_Score": 8.5,
        "Feedback": "Lập luận tốt, nhưng tính nhầm ở bước cuối cùng."
    }}
    """
    
    try:
        # Dùng temperature=0.0 để AI chấm điểm nhất quán, không bị cảm tính
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        
        # Xử lý chuỗi văn bản trả về để đọc JSON
        result_text = response.text.strip()
        
        # Dọn dẹp nếu LLM lỡ sinh ra block code markdown (```json ... ```)
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        # Chuyển chuỗi thành Dictionary trong Python
        data = json.loads(result_text)
        
        # Trả về 4 giá trị để điền vào 4 cột
        return pd.Series([
            data.get("Logic_Score", 0), 
            data.get("Math_Score", 0), 
            data.get("Average_Score", 0), 
            data.get("Feedback", "Không có nhận xét")
        ])
        
    except Exception as e:
        print(f"Lỗi API hoặc lỗi phân tích JSON: {e}")
        # Nếu có lỗi (như rớt mạng, vượt quá quota), điền 0 và báo lỗi
        return pd.Series([0, 0, 0, f"Lỗi: {str(e)}"])

# 3. Đọc dữ liệu từ file
df = pd.read_csv("test1_results.csv")

# Tớ đang để cắt 5 dòng đầu tiên để test cho nhanh. 
# Khi nào chạy ngon, cậu xóa chữ `.head(5)` này đi để chấm toàn bộ 244 câu nhé.
df_test = df.head(5).copy()

print("Bắt đầu gọi API nhờ AI chấm điểm...")

# 4. Áp dụng hàm đánh giá và tạo 4 cột mới
df_test[['Logic_Score', 'Math_Score', 'Average_Score', 'Judge_Feedback']] = df_test.apply(
    lambda row: evaluate_with_llm(row['Latex Input'], row['model_full_answer']), axis=1
)

# 5. Lưu kết quả ra file mới
output_file = "llm_api_judged_results.csv"
df_test.to_csv(output_file, index=False)
print(f"Đã chấm xong! Kết quả siêu chuẩn được lưu tại: {output_file}")