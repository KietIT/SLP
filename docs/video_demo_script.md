# Kịch bản video demo Vietnamese ASR dưới nhiễu

## Mục tiêu và phạm vi

Video dài khoảng **6 phút** và minh họa ba cấu hình trên cùng một tín hiệu đầu
vào:

- Ordinary LoRA (`lambda = 0`)
- Tone-aware LoRA (`lambda = 0.05`, selected method)
- Tone-aware LoRA (`lambda = 0.1`, locked control)

Ứng dụng nhận âm thanh bằng hai cách: thu trực tiếp từ microphone hoặc tải tệp
WAV lên. Model thực hiện **nhận dạng tiếng nói tiếng Việt**, không dự đoán loại
nhiễu hay SNR. Tùy chọn trộn nhiễu trong giao diện chỉ tạo đầu vào có kiểm soát
để so sánh ba model một cách công bằng.

Nếu nhập câu tham chiếu, ứng dụng hiển thị WER, CER, TER, DER, FCER, SWDR và
Substitution/Deletion/Insertion. Nếu không có tham chiếu, ứng dụng chỉ hiển thị
transcript và latency; không được diễn giải transcript đó như một phép đo định
lượng.

## Chuẩn bị trước khi quay

1. Dùng phòng yên tĩnh, đóng các ứng dụng đang dùng GPU và cắm nguồn máy tính.
2. Khởi động ứng dụng từ root của repository:

   ```powershell
   conda activate slp
   python -m streamlit run scripts/demo_app.py
   ```

3. Mở URL localhost mà ứng dụng in ra. Cho phép trình duyệt dùng microphone.
4. Chạy thử một WAV ngắn để ba checkpoint được load và làm nóng GPU trước khi
   quay. Không tính lần chạy này vào latency trình bày.
5. Chuẩn bị sẵn một WAV clean dự phòng dài dưới 15 giây. Giữ nguyên câu tham
   chiếu có dấu, mã hóa UTF-8.
6. Dùng cùng một bản ghi gốc cho clean, 10 dB và 0 dB. Không thu lại riêng từng
   điều kiện vì nội dung, tốc độ và âm lượng giọng sẽ thay đổi.
7. Đóng mọi cửa sổ có token, đường dẫn cá nhân hoặc thông tin tài khoản trước
   khi quay màn hình.

## Câu đọc đề xuất

Câu chính, chứa năm từ ngắn trong danh sách phân tích deletion:

> Đã có một cô bé là một học sinh ngoan và rất chăm chỉ.

Câu phụ, giàu đối lập thanh điệu:

> Bà bảo bé Bảo mang bốn quả bưởi về nhà.

Đọc tự nhiên, cách microphone khoảng 15–20 cm. Không cố phát âm quá chậm hoặc
đánh vần từng tiếng.

## Kịch bản quay 5–7 phút

### 0:00–0:35 — Giới thiệu

**Thao tác:** mở slide tiêu đề hoặc trang đầu của ứng dụng.

**Lời nói gợi ý:**

> Đây là demo nhận dạng tiếng nói tiếng Việt dưới nhiễu. Chúng tôi so sánh
> Ordinary LoRA với hai cấu hình tone-aware lambda 0.05 và 0.1. Cả ba model nhận
> cùng một audio; model không phát hiện noise type, mà chỉ sinh transcript.

### 0:35–1:10 — Giới thiệu giao diện

**Thao tác:** chỉ lần lượt hai input `Microphone` và `Upload WAV`, ô reference,
tùy chọn noise/SNR và bảng kết quả.

**Lời nói gợi ý:**

> Có thể thu trực tiếp hoặc tải WAV. Ứng dụng chuẩn hóa audio về mono 16 kHz và
> giới hạn 15 giây. Khi có reference, hệ thống tính sáu metric aligned cùng số
> lỗi substitution, deletion và insertion.

### 1:10–2:00 — Thu trực tiếp hoặc tải WAV clean

**Thao tác:** nhập nguyên văn câu chính vào ô reference, thu câu bằng microphone
rồi nghe lại. Nếu microphone không hoạt động, tải WAV dự phòng và nói rõ đây là
cùng câu đã thu trước.

**Lời nói gợi ý:**

> Tôi dùng một bản ghi duy nhất làm nguồn. Trước hết ta chạy bản clean. Kết quả
> của từng model được đặt cạnh nhau nên không cần suy luận từ ba lần thu khác
> nhau.

### 2:00–2:50 — Kết quả clean

**Thao tác:** chọn `clean`, chạy inference và phóng to bảng transcript/metric.

**Lời nói gợi ý:**

> Đây là transcript trên tín hiệu clean. Các giá trị này chỉ mô tả câu demo;
> kết luận của nghiên cứu dựa trên toàn bộ benchmark 2.300 mẫu và bootstrap,
> không dựa trên một câu thuận lợi.

Đọc đúng các số đang hiển thị; không đọc số từ kịch bản hoặc điền trước kết quả.

### 2:50–3:40 — Controlled noise 10 dB

**Thao tác:** giữ nguyên audio/reference, chọn một noise có sẵn và SNR `10 dB`.
Phát audio đã trộn khoảng 3–5 giây rồi chạy inference.

**Lời nói gợi ý:**

> Bây giờ ứng dụng trộn noise vào đúng bản ghi vừa dùng ở mức 10 dB. Vì clean
> speech không đổi, khác biệt transcript phản ánh khả năng chịu nhiễu của ba
> cấu hình trên cùng đầu vào.

### 3:40–4:30 — Controlled noise 0 dB

**Thao tác:** chỉ đổi SNR thành `0 dB`, phát một đoạn ngắn và chạy lại.

**Lời nói gợi ý:**

> Ở 0 dB, công suất noise xấp xỉ công suất speech nên đây là điều kiện khó.
> Chúng ta xem đồng thời WER và các metric tiếng Việt, không chọn model chỉ dựa
> trên một ô có giá trị thấp hơn.

### 4:30–5:10 — Real-world noise

**Thao tác:** bật quạt hoặc tạo tiếng nói nền, thu câu phụ bằng microphone, chọn
`không trộn thêm noise`, nhập reference rồi chạy.

**Lời nói gợi ý:**

> Đây là ví dụ định tính trong môi trường thật. SNR và loại nhiễu không được
> kiểm soát, nên kết quả này minh họa cách sử dụng chứ không thay thế thí nghiệm
> benchmark.

### 5:10–5:50 — Liên hệ kết quả toàn bộ benchmark

**Thao tác:** chuyển sang slide kết quả chính đã kiểm định.

**Lời nói gợi ý:**

> Trên VIVOS final, WER của Ordinary là 13.1073%, lambda 0.05 là 12.9914% và
> lambda 0.1 là 13.1288%. Trên FLEURS clean, Ordinary có WER tốt nhất. Toàn bộ
> khoảng tin cậy paired bootstrap hiện chứa 0, vì vậy chúng tôi không tuyên bố
> tone-aware vượt trội có ý nghĩa thống kê.

### 5:50–6:15 — Kết luận

**Thao tác:** hiển thị cùng lúc ba transcript cuối và trang kết luận.

**Lời nói gợi ý:**

> Demo cho thấy pipeline có thể xử lý cả WAV và microphone, tạo noise có kiểm
> soát, và phân tích lỗi bằng đúng metric của paper. Kết quả hiện tại cho thấy
> tone supervision tạo trade-off nhỏ phụ thuộc điều kiện và metric; multi-seed
> cùng các ablation là bước kiểm định tiếp theo.

## Phương án dự phòng khi quay

- **Trình duyệt không thu được microphone:** tải WAV dự phòng; microphone trên
  localhost thường dùng được nhưng có thể bị chặn bởi quyền riêng tư của Windows.
- **Không có noise preset:** chỉ demo clean và real-world noise; không tự gán nhãn
  SNR cho bản thu thực tế.
- **GPU hết bộ nhớ:** đóng job khác, khởi động lại ứng dụng và chạy từng model nếu
  giao diện có chế độ tuần tự. Không đổi checkpoint hoặc model size.
- **Inference lần đầu chậm:** làm nóng đủ ba checkpoint trước khi bấm quay và vẫn
  báo latency hiển thị là latency của demo, không phải benchmark throughput.
- **Transcript live bất lợi:** giữ nguyên kết quả. Có thể dùng thêm WAV benchmark
  đã chốt, nhưng phải ghi rõ nguồn; không thu đi thu lại để chọn một take đẹp.
- **Không có reference:** chỉ trình bày transcript/latency và bỏ qua nhận xét về
  WER, TER hoặc các loại lỗi.
- **Ứng dụng lỗi giữa video:** dùng screen recording của lượt chạy thử làm B-roll,
  sau đó quay lại phần kết luận; không sửa CSV/JSON kết quả bằng tay.

## Checklist trước khi nộp video

- [ ] Video dài từ 5 đến 7 phút, chữ trong bảng đọc được ở 1080p.
- [ ] Cả ba cấu hình dùng đúng một audio cho mỗi phép so sánh.
- [ ] Clean, controlled noise và real-world noise được gọi tên chính xác.
- [ ] Không gọi ứng dụng là noise detector.
- [ ] Không tuyên bố một ví dụ đơn lẻ là bằng chứng thống kê.
- [ ] Không tuyên bố tone-aware vượt trội hoặc SOTA.
- [ ] Có nhắc final benchmark, FLEURS và kết luận bootstrap.
- [ ] Không lộ đường dẫn cá nhân, token hoặc thông tin tài khoản.
