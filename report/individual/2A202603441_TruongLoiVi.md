# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603441**
- **Họ và tên: Trương Lợi Vĩ**
- **SEED** (in ở ô đầu notebook): 703

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | yes |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | yes |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | yes |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.771      0.871        0.824   
2. BoW + MLP (deep learning)         1.000   0.880      0.936        0.919   
3a. Encoder (MiniLM) + kNN           0.841   0.892      0.865        0.923   
3b. Encoder (MiniLM) + LogReg        0.927   0.916      0.921        0.947   
3c. Encoder (MiniLM) + MLP head      0.939   0.928      0.933        0.947   
4a. LLM nhỏ zero-shot                0.909   1.000      0.952        0.952   
4c. LLM nhỏ few-shot                 0.952   1.000      0.976        0.976   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.759      32664.115       0.046   
2. BoW + MLP (deep learning)         6.748      43539.857       0.322   
3a. Encoder (MiniLM) + kNN           3.652        164.409      14.401   
3b. Encoder (MiniLM) + LogReg        3.984       1489.131       6.576   
3c. Encoder (MiniLM) + MLP head      4.703       2043.502       6.711   
4a. LLM nhỏ zero-shot                0.000          1.122     751.778   
4c. LLM nhỏ few-shot                 0.000          1.195     753.989   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.073         0.000  
2. BoW + MLP (deep learning)          0.430         0.000  
3a. Encoder (MiniLM) + kNN           17.968         0.000  
3b. Encoder (MiniLM) + LogReg         7.710         0.000  
3c. Encoder (MiniLM) + MLP head       7.883         0.000  
4a. LLM nhỏ zero-shot              1766.249         0.033  
4c. LLM nhỏ few-shot               1387.970         0.076  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

### 3 dự đoán lệch nhiều nhất so với kết quả

**1. Thí nghiệm Embedding – So sánh BoW và Encoder + Logistic Regression**

- **Dự đoán:** Encoder + Logistic Regression sẽ phân loại đúng nhiều hơn BoW nhờ khả năng hiểu ngữ nghĩa thay vì chỉ dựa vào từ khóa.
- **Kết quả:** Cả hai mô hình đều phân loại đúng 5/6 tin nhắn.
- **Vì sao lệch:** Embedding có khả năng nắm bắt ngữ nghĩa tốt hơn nhưng không đảm bảo luôn vượt trội trên tập dữ liệu nhỏ. Một số tin Spam có cách diễn đạt quen thuộc vẫn được BoW nhận diện tốt nhờ các từ khóa đặc trưng.

**2. Thí nghiệm DL – Bỏ hàm ReLU**

- **Dự đoán:** F1 khoảng 0.900.
- **Kết quả:** F1 thực tế đạt 0.936, bằng với MLP có ReLU khi làm tròn 3 chữ số.
- **Vì sao lệch:** Hai lớp Linear liên tiếp không có hàm kích hoạt vẫn tương đương một phép biến đổi tuyến tính. Trên bài toán này, mô hình tuyến tính vẫn phân loại khá tốt nên việc bỏ ReLU không làm F1 giảm như kỳ vọng.

**3. Stress Test – Khả năng tổng quát hóa trên dữ liệu khó**

- **Dự đoán:** Accuracy của các mô hình trên 50 case khó có thể giảm xuống khoảng 60–75%; LLM có thể bị đánh lừa bởi Prompt Injection.
- **Kết quả:** BoW + Logistic Regression đạt Accuracy 54%, BoW + MLP đạt 58%, trong khi LLM Zero-shot và LLM với prompt phòng thủ đều đạt 100%.
- **Vì sao lệch:** BoW và MLP phụ thuộc nhiều vào đặc trưng từ vựng nên dễ bị đánh lừa bởi cách diễn đạt và ngữ cảnh mới. Ngược lại, LLM thể hiện khả năng hiểu ngữ cảnh tốt hơn và xử lý được các chỉ dẫn giả mạo trong bộ test. Tuy nhiên, 50 case chưa đủ để khẳng định LLM luôn chính xác trong thực tế.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

BoW + Logistic Regression là mô hình tụt nhiều nhất, với F1 giảm từ 0.824 trên tập test thông thường xuống 0.378 trên 50 case khó, tương đương giảm 0.446 điểm. Các nhóm khó nhất là Thao túng/lừa đảo (12%), Né từ khóa (17%) và Spam giọng tự nhiên (20%).

Ví dụ thứ nhất là tin nhắn giả dạng người quen giới thiệu một công việc kinh doanh mỹ phẩm và đề nghị gửi catalogue. Mô hình BoW dễ nhầm đây là tin nhắn cá nhân vì nội dung sử dụng ngôn ngữ thân mật và thiếu các từ khóa spam điển hình. Ví dụ thứ hai là tin nhắn giả danh người mẹ bị mất điện thoại và yêu cầu chuyển £200. Cách diễn đạt giống hội thoại gia đình khiến mô hình khó nhận diện ý đồ lừa đảo.

Nguyên nhân chính là BoW chỉ dựa trên sự xuất hiện của các từ riêng lẻ, không hiểu đầy đủ ngữ cảnh, mối quan hệ giữa các câu và mục đích của người gửi. Vì vậy, mô hình hoạt động tốt trên dữ liệu thông thường nhưng giảm hiệu suất đáng kể khi gặp các tình huống được thiết kế để đánh lừa.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 5. Thử thách cá nhân

**Đề bài (SEED = 703 – Đề số 2):** Giảm Vocabulary còn 100 từ và tăng lên 5.000 từ bằng `build_vocab(..., max_size)`. Huấn luyện lại mô hình Logistic Regression và so sánh F1 trên tập test.

**Cách thực hiện:**

- Tạo hai bộ từ điển với kích thước lần lượt là 100 và 5.000 từ từ tập huấn luyện.
- Vector hóa dữ liệu train và test theo từng bộ từ điển.
- Huấn luyện lại mô hình Logistic Regression với `lr = 0.5`, `epochs = 300`.
- Sử dụng ngưỡng 0.5 để phân loại và đánh giá bằng Precision, Recall, F1.
- Ghi nhận thời gian huấn luyện của từng cấu hình.

**Kết quả thực nghiệm:**

| Chỉ số | Vocabulary 100 | Vocabulary 5.000 |
|---|---:|---:|
| Số từ thực tế | 100 | 5.000 |
| F1 | 0.797 | **0.863** |
| Precision | 0.950 | **1.000** |
| Recall | 0.687 | **0.759** |
| Thời gian train | 0.117 giây | 7.458 giây |

**Giải thích:**

Vocabulary 5.000 từ đạt F1 = 0.863, cao hơn so với 0.797 của Vocabulary 100. Precision tăng từ 0.950 lên 1.000 và Recall tăng từ 0.687 lên 0.759. Nguyên nhân là Vocabulary lớn giữ lại nhiều đặc trưng từ vựng hơn, giúp mô hình nhận diện được nhiều dấu hiệu Spam và giảm số trường hợp bỏ sót.

Tuy nhiên, thời gian huấn luyện tăng từ 0.117 giây lên 7.458 giây, tức khoảng 63.7 lần. Vocabulary lớn làm tăng số chiều của dữ liệu đầu vào, khiến quá trình tính toán và cập nhật trọng số tốn nhiều thời gian hơn.

**Kết luận:** Tăng kích thước Vocabulary giúp cải thiện chất lượng phân loại trong thí nghiệm này, nhưng đồng thời làm tăng đáng kể thời gian huấn luyện. Vì vậy, cần lựa chọn kích thước Vocabulary phù hợp với sự cân bằng giữa hiệu quả phân loại và tài nguyên tính toán.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 6. Phương án cho sản phẩm thật

### 1. Lựa chọn mô hình

Với yêu cầu xử lý 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây) và độ trễ p95 dưới 50 ms/tin, các phương pháp đáp ứng SLA trong thực nghiệm gồm:

- BoW + Logistic Regression: p95 = 0.073 ms, F1 (SUB) = 0.824.
- BoW + MLP: p95 = 0.430 ms, F1 (SUB) = 0.919.
- Embedding + kNN: p95 = 17.968 ms, F1 (SUB) = 0.923.
- Embedding + Logistic Regression: p95 = 7.710 ms, F1 (SUB) = 0.947.
- Embedding + MLP: p95 = 7.883 ms, F1 (SUB) = 0.947.

**Lựa chọn: Embedding (MiniLM) + Logistic Regression.**

Mô hình đạt F1 (SUB) = 0.947, p95 = 7.710 ms/tin, thấp hơn đáng kể so với giới hạn 50 ms. Throughput khi xử lý theo lô đạt khoảng 1.489 tin/giây, cao hơn mức trung bình 116 tin/giây cần thiết. Chi phí inference không phát sinh phí API trong thí nghiệm.

LLM nhỏ Few-shot đạt F1 (SUB) cao hơn là 0.976, nhưng p95 lên đến 1.387,970 ms/tin và chi phí khoảng 0.076 USD/1.000 tin. Nếu xử lý toàn bộ 10 triệu tin/ngày trong 30 ngày, chi phí ước tính khoảng 22.800 USD/tháng. Vì vậy, LLM không phù hợp làm mô hình xử lý chính trong hệ thống yêu cầu độ trễ thấp.

### 2. Đánh đổi được chấp nhận

Phương án Embedding + Logistic Regression có F1 thấp hơn LLM Few-shot, nhưng đổi lại có độ trễ thấp, không phát sinh chi phí API theo số lượng tin nhắn và đáp ứng SLA. Việc lựa chọn này chấp nhận đánh đổi một phần chất lượng phân loại để đảm bảo khả năng xử lý khối lượng lớn và tính ổn định của hệ thống.

### 3. Thiết kế hệ thống lai

Có thể xây dựng hệ thống kết hợp Embedding + Logistic Regression và một mô hình phụ để xử lý các trường hợp không chắc chắn:

- Tin có xác suất phân loại rõ ràng: xử lý trực tiếp bằng Embedding + Logistic Regression.
- Tin có xác suất gần ngưỡng quyết định: chuyển sang mô hình phụ đã được triển khai cục bộ để kiểm tra thêm.
- Các trường hợp nghi ngờ hoặc có dấu hiệu lừa đảo: đưa vào hàng đợi để phân tích chuyên sâu bằng LLM, không chặn luồng xử lý đồng bộ.

Thiết kế này tận dụng tốc độ của mô hình chính và khả năng hiểu ngữ cảnh của mô hình phụ, đồng thời hạn chế chi phí khi không phải sử dụng LLM cho mọi tin nhắn.

**Kết luận:** Embedding + Logistic Regression là lựa chọn phù hợp nhất trong thí nghiệm cho yêu cầu sản phẩm đã đặt ra. Mô hình đáp ứng giới hạn độ trễ, đạt F1 cao trong nhóm phương pháp đáp ứng SLA và không phát sinh chi phí API. Trước khi triển khai thực tế, cần tiếp tục kiểm thử trên dữ liệu SMS quy mô lớn và các tình huống spam mới để đánh giá độ ổn định.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

## 7. AI log và điều đã học

### 7.1. AI log

| Nội dung trao đổi | Kết quả AI hỗ trợ | Điều chỉnh và kiểm tra |
|---|---|---|
| Giải thích Logistic Regression, Gradient Descent và BoW | Giải thích cách chuyển văn bản thành vector, tính xác suất và cập nhật trọng số | Tự triển khai code bằng NumPy, không sao chép code AI |
| Giải thích MLP và hiện tượng Overfitting | Làm rõ vai trò của hidden layer, ReLU và validation loss | Đối chiếu với biểu đồ loss và kết quả thực nghiệm trong notebook |
| Giải thích Embedding và Cosine Similarity | Làm rõ cách biểu diễn câu thành vector và so sánh mức độ tương đồng | Đối chiếu với các trường hợp kiểm tra trong notebook |
| Hỗ trợ triển khai Phần 4 – LLM | Hỗ trợ xây dựng hàm gọi API, prompt phân loại, xử lý nhãn và gọi mô hình theo batch | Kiểm tra lại tham số API, xử lý nhãn không hợp lệ và thử nghiệm với Gemini |
| Khắc phục lỗi khi gọi Gemini API | Giải thích lỗi giới hạn tốc độ 429 và lỗi tham số `reasoning_effort` | Điều chỉnh thời gian retry và tham số suy luận để tiếp tục chạy thử nghiệm |
| Phân tích kết quả Stress Test và so sánh mô hình | Hỗ trợ diễn giải F1, độ chính xác theo từng nhóm case và nguyên nhân mô hình phân loại sai | Đối chiếu với bảng kết quả thực tế, không suy diễn những trường hợp chưa có dự đoán chi tiết |
| Hỗ trợ trình bày báo cáo | Gợi ý cách diễn giải kết quả, phân tích đánh đổi giữa chất lượng, tốc độ và chi phí | Điều chỉnh nội dung theo số liệu thực nghiệm trong notebook |

### 7.2. Điều đã học

Qua bài thực hành, có thể hiểu rõ hơn rằng chất lượng mô hình không chỉ phụ thuộc vào thuật toán mà còn phụ thuộc vào cách biểu diễn dữ liệu. BoW hoạt động tốt với những từ khóa quen thuộc nhưng gặp khó khăn khi spam sử dụng ngôn ngữ tự nhiên hoặc cố tình né từ khóa; Embedding có khả năng nắm bắt ngữ nghĩa tốt hơn.

Bên cạnh đó, mô hình có F1 cao chưa chắc phù hợp để triển khai thực tế. Cần đánh giá đồng thời độ trễ, throughput và chi phí để lựa chọn giải pháp đáp ứng yêu cầu sản phẩm. Đặc biệt, LLM có thể đạt kết quả phân loại tốt nhưng không phải lúc nào cũng là lựa chọn tối ưu khi phải xử lý hàng triệu tin nhắn mỗi ngày.
